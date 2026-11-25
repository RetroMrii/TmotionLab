import argparse
import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ml.src.data.goemotions_dataset import load_goemotions_rows

ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / "ml/configs/synthetic_goemotions_large_scale.json"

CANDIDATES = ROOT / "data/synthetic/goemotions_large_scale/candidates.jsonl"

QA_DIR = ROOT / "data/synthetic/goemotions_large_scale/quality"

REPORT = QA_DIR / "integrity_audit.json"
JOURNAL = QA_DIR / "blind_annotations.jsonl"
DECISIONS = QA_DIR / "quality_decisions.jsonl"

FIXTURE = QA_DIR / "fixture_candidates.jsonl"

LABEL_MANIFEST = ROOT / "data/processed/goemotions_v001/manifest.json"

VERIFY_PROMPT = """
You are an independent annotator evaluating an English-language
conversational comment under the GoEmotions 28-label taxonomy.

You have not generated this comment. Evaluate only its actual
wording and reasonably inferable conversational context.

Assign all emotions independently supported by the comment.
Do not assume that each comment has a particular target emotion.

Important distinctions:
- Approval endorses an action; admiration appreciates merits.
- Annoyance is irritation; anger is stronger indignation.
- Disapproval judges negatively; disappointment reflects unmet expectations.
- Confusion is lack of understanding; curiosity seeks information.
- Realization recognizes a new connection or fact.
- Nervousness involves anticipatory unease; fear involves perceived danger.
- Relief involves reduction of prior concern; joy is happiness.
- Grief relates to meaningful loss; sadness is broader sorrow.
- Remorse concerns one's own actions.
- Pride concerns achievement or identity, often one's own or a close person's.

Do not assign labels based solely on the topic.
Do not infer the speaker's emotions from someone else's feelings.
Do not force multi-label annotations.
Neutral is not mechanically exclusive with other labels.

A comment may be ambiguous, unnatural, or insufficiently
informative. Flag these cases rather than inventing certainty.

Return all supported labels using the exact allowed vocabulary.
Provide brief textual evidence for your assignments.
Flag cases needing human review.

Do not rewrite the comment.
Do not infer or guess the generation target.
"""


class Annotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labels: list[str]
    evidence: list[str]
    ambiguous: bool
    unnatural: bool
    needs_human_review: bool


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)

    temp = path.with_suffix(path.suffix + ".tmp")

    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    os.replace(temp, path)


def label_names():
    manifest = json.loads(LABEL_MANIFEST.read_text(encoding="utf-8"))

    names = manifest["label_names"]

    if len(names) != 28 or len(set(names)) != 28:
        raise ValueError("Unexpected label vocabulary")

    return names


def source_texts():
    result = {}

    for split in ("train", "validation", "test"):
        rows, _ = load_goemotions_rows(split)
        result[split] = {normalize(row["text"]) for row in rows}

    return result


def validate_candidates(rows, names):
    allowed = set(names)
    ids = set()

    for row in rows:
        cid = row["candidate_id"]

        if not isinstance(cid, str) or not cid:
            raise ValueError("Invalid candidate ID")

        if cid in ids:
            raise ValueError(f"Duplicate candidate ID: {cid}")

        ids.add(cid)

        text = row["text"]

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty text: {cid}")

        primary = row["intended_primary_label"]
        secondary = row["proposed_secondary_labels"]

        if primary not in allowed:
            raise ValueError(f"Unknown primary: {cid}")

        if not isinstance(secondary, list):
            raise ValueError(f"Invalid secondary labels: {cid}")

        if len(secondary) != len(set(secondary)):
            raise ValueError(f"Duplicate secondary labels: {cid}")

        if primary in secondary:
            raise ValueError(f"Primary repeated: {cid}")

        if not set(secondary).issubset(allowed):
            raise ValueError(f"Unknown secondary label: {cid}")

        if row.get("approval_status") != "pending":
            raise ValueError(f"Unexpected approval status: {cid}")

        if row.get("label_status") != "unverified":
            raise ValueError(f"Unexpected label status: {cid}")


def audit(rows, names):
    validate_candidates(rows, names)

    real = source_texts()
    normalized_groups = defaultdict(list)
    flags = {}

    for row in rows:
        cid = row["candidate_id"]
        normalized = normalize(row["text"])

        normalized_groups[normalized].append(cid)

        problems = []

        if len(row["text"].strip()) < 12:
            problems.append("very_short_text")

        if "\ufffd" in row["text"]:
            problems.append("replacement_character")

        for split, texts in real.items():
            if normalized in texts:
                problems.append(f"exact_overlap_{split}")

        flags[cid] = problems

    for ids in normalized_groups.values():
        if len(ids) > 1:
            for cid in ids:
                flags[cid].append("duplicate_candidate_text")

    counts = Counter()
    primary_counts = Counter()

    for row in rows:
        cid = row["candidate_id"]

        primary_counts[row["intended_primary_label"]] += 1
        counts.update(flags[cid])

    result = {
        "experiment": "OPT-009",
        "stage": "integrity_audit",
        "candidate_count": len(rows),
        "primary_label_counts": dict(primary_counts),
        "flag_counts": dict(counts),
        "flagged_candidates": sum(bool(value) for value in flags.values()),
        "exact_duplicate_groups": sum(
            len(ids) > 1 for ids in normalized_groups.values()
        ),
        "notes": [
            "Exact normalized text comparisons only.",
            "No semantic or near-duplicate detection yet.",
            "Validation/test text used for exclusion only.",
            "Automated integrity checks do not verify labels.",
            "No candidates automatically approved.",
        ],
    }

    write_json(REPORT, result)

    return flags, result


def verify(rows, names, flags, args):
    from openai import OpenAI

    if not args.model or args.model.startswith("SET_"):
        raise ValueError("Set an actual verifier model ID")

    if not args.max_calls or args.max_calls <= 0:
        raise ValueError("--max-calls must be positive")

    if args.max_usd is None or args.max_usd <= 0:
        raise ValueError("--max-usd must be positive")

    if args.input_per_million is None or args.output_per_million is None:
        raise ValueError("Specify API prices explicitly")

    if (
        min(
            args.input_per_million,
            args.output_per_million,
        )
        < 0
    ):
        raise ValueError("Invalid API prices")

    if args.request_reserve_usd is None or args.request_reserve_usd <= 0:
        raise ValueError("Specify a positive request cost reserve")

    previous = read_jsonl(JOURNAL) if JOURNAL.exists() else []

    completed = {}

    for record in previous:
        cid = record["candidate_id"]

        if cid in completed:
            raise ValueError(f"Duplicate journal record: {cid}")

        completed[cid] = record

    candidates_by_id = {row["candidate_id"]: row for row in rows}

    for cid, record in completed.items():
        row = candidates_by_id.get(cid)

        if row is None:
            raise ValueError(f"Journal candidate missing: {cid}")

        if record["text_sha256"] != digest(row["text"]):
            raise ValueError(f"Candidate text changed: {cid}")

        if record["verifier_model_requested"] != args.model:
            raise ValueError("Existing annotations use a different verifier model")

    spent = sum(float(record["estimated_cost_usd"]) for record in previous)

    remaining = [
        row
        for row in rows
        if row["candidate_id"] not in completed
        and not (set(flags[row["candidate_id"]]) - SEMANTIC_REVIEW_FLAGS)
    ]

    client = OpenAI(max_retries=2, timeout=120)
    calls = 0

    for row in remaining:
        if calls >= args.max_calls:
            break

        # Conservative budget reservation before sending a call.
        if spent + args.request_reserve_usd > args.max_usd:
            print("[STOP] Budget reserve reached")
            break

        text = row["text"]

        # The verifier receives ONLY text and label vocabulary.
        # No primary label, secondary labels, or slot specification.
        user_content = "Allowed labels:\n" + ", ".join(names) + "\n\nComment:\n" + text

        response = client.responses.parse(
            model=args.model,
            instructions=VERIFY_PROMPT,
            input=[
                {
                    "role": "user",
                    "content": user_content,
                }
            ],
            text_format=Annotation,
            store=False,
        )

        parsed = response.output_parsed

        if parsed is None:
            raise RuntimeError("No parsed annotation for " + row["candidate_id"])

        if len(parsed.labels) != len(set(parsed.labels)):
            raise ValueError("Duplicate verifier labels")

        if not set(parsed.labels).issubset(names):
            raise ValueError("Verifier returned unknown labels")

        usage = response.usage
        if usage is None:
            raise RuntimeError("Missing API usage; cannot track spending")

        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        cost = (
            input_tokens * args.input_per_million
            + output_tokens * args.output_per_million
        ) / 1_000_000

        record = {
            "candidate_id": row["candidate_id"],
            "text_sha256": digest(text),
            "verifier_model_requested": args.model,
            "verifier_model_returned": response.model,
            "response_id": response.id,
            "annotation": parsed.model_dump(mode="json"),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": cost,
            "annotation_status": "machine_proposed",
        }

        append_jsonl(JOURNAL, record)

        if cost > args.request_reserve_usd:
            print(
                "[WARNING] Actual request cost exceeded the "
                "configured reserve. Stopping after recording usage."
            )
            spent += cost
            calls += 1
            break

        spent += cost
        calls += 1

        print(
            f"[VERIFIED BY MODEL] {row['candidate_id']} "
            f"calls={calls} estimated_spend=${spent:.4f}"
        )

    print(f"New API calls: {calls}")
    print(f"Total estimated spend: ${spent:.4f}")
    print(f"Journal: {JOURNAL}")


SEMANTIC_REVIEW_FLAGS = {
    "synthetic_semantic_similarity",
    "real_data_semantic_similarity",
}


def merge_semantic_flags(rows, flags, semantic_dir, input_path):
    if semantic_dir is None:
        return flags

    report_path = semantic_dir / "report.json"
    flag_path = semantic_dir / "candidate_flags.jsonl"

    report = json.loads(report_path.read_text(encoding="utf-8"))

    actual_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()

    if report["candidate_file_sha256"] != actual_hash:
        raise ValueError("Semantic audit was run on a different candidate file")

    if report["candidate_count"] != len(rows):
        raise ValueError("Semantic candidate count mismatch")

    expected_ids = {row["candidate_id"] for row in rows}
    semantic_rows = read_jsonl(flag_path)

    if len(semantic_rows) != len(rows):
        raise ValueError("Incomplete semantic flag report")

    merged = {cid: list(values) for cid, values in flags.items()}
    seen = set()

    for record in semantic_rows:
        cid = record["candidate_id"]

        if cid not in expected_ids or cid in seen:
            raise ValueError(f"Invalid semantic candidate ID: {cid}")

        seen.add(cid)

        reported = record["flags"]

        if not isinstance(reported, list):
            raise ValueError(f"Invalid semantic flags: {cid}")

        if not set(reported).issubset(SEMANTIC_REVIEW_FLAGS):
            raise ValueError(f"Unexpected semantic flag: {cid}")

        merged[cid] = sorted(set(merged[cid]) | set(reported))

    if seen != expected_ids:
        raise ValueError("Missing semantic audit candidates")

    return merged


def decisions(rows, flags):
    annotations = {}

    if JOURNAL.exists():
        for record in read_jsonl(JOURNAL):
            cid = record["candidate_id"]

            if cid in annotations:
                raise ValueError(f"Duplicate annotation: {cid}")

            annotations[cid] = record

    counts = Counter()
    output = []

    candidate_ids = {row["candidate_id"] for row in rows}

    if not set(annotations).issubset(candidate_ids):
        raise ValueError("Annotation journal contains unknown candidates")

    for row in rows:
        cid = row["candidate_id"]

        proposed = {
            row["intended_primary_label"],
            *row["proposed_secondary_labels"],
        }

        verified = annotations.get(cid)

        all_flags = set(flags[cid])

        hard_flags = all_flags - SEMANTIC_REVIEW_FLAGS
        review_flags = all_flags & SEMANTIC_REVIEW_FLAGS

        reasons = sorted(all_flags)

        if verified is not None:
            if verified["text_sha256"] != digest(row["text"]):
                raise ValueError(f"Annotation text mismatch: {cid}")

        if hard_flags:
            status = "rejected_integrity"

        elif verified is None:
            status = "needs_review" if review_flags else "pending_annotation"

        else:
            annotation = verified["annotation"]
            independent = set(annotation["labels"])

            if annotation["unnatural"]:
                reasons.append("unnatural_text")

            if annotation["ambiguous"]:
                reasons.append("ambiguous_annotation")

            if annotation["needs_human_review"]:
                reasons.append("verifier_requested_review")

            if row["intended_primary_label"] not in independent:
                reasons.append("primary_not_supported_by_verifier")

            if independent != proposed:
                reasons.append("label_disagreement")

            reasons = sorted(set(reasons))

            status = "needs_review" if reasons else "provisional_agreement"

        counts[status] += 1

        output.append(
            {
                "candidate_id": cid,
                "status": status,
                "reasons": reasons,
                "generator_labels": sorted(proposed),
                "verifier_labels": (
                    verified["annotation"]["labels"] if verified is not None else None
                ),
                "approved_for_training": False,
            }
        )

    DECISIONS.parent.mkdir(parents=True, exist_ok=True)

    temp = DECISIONS.with_suffix(".jsonl.tmp")

    with temp.open("w", encoding="utf-8") as stream:
        for record in output:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    os.replace(temp, DECISIONS)

    return counts


def make_fixture():
    QA_DIR.mkdir(parents=True, exist_ok=True)

    if FIXTURE.exists():
        raise FileExistsError(FIXTURE)

    examples = [
        ("relief", "The missing keys were in the other bag all along."),
        ("realization", "Oh, that explains why the screen kept freezing."),
        ("gratitude", "Thanks for staying late to help me finish this."),
        ("neutral", "The next train leaves from platform four."),
    ]

    with FIXTURE.open("x", encoding="utf-8") as stream:
        for index, (primary, text) in enumerate(examples):
            row = {
                "candidate_id": f"fixture_{index:03d}",
                "text": text,
                "intended_primary_label": primary,
                "proposed_secondary_labels": [],
                "approval_status": "pending",
                "label_status": "unverified",
            }
            stream.write(json.dumps(row) + "\n")

    print(f"Fixture created: {FIXTURE}")


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fixture", action="store_true")
    mode.add_argument("--audit", action="store_true")
    mode.add_argument("--verify", action="store_true")

    parser.add_argument("--input", type=Path, default=CANDIDATES)
    parser.add_argument("--model")
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-usd", type=float)
    parser.add_argument("--request-reserve-usd", type=float)
    parser.add_argument("--input-per-million", type=float)
    parser.add_argument("--output-per-million", type=float)
    parser.add_argument(
        "--semantic-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.fixture:
        make_fixture()
        return

    global REPORT, JOURNAL, DECISIONS

    if args.input.resolve() != CANDIDATES.resolve():
        fixture_output = QA_DIR / "fixture_results" / args.input.stem

        REPORT = fixture_output / "integrity_audit.json"
        JOURNAL = fixture_output / "blind_annotations.jsonl"
        DECISIONS = fixture_output / "quality_decisions.jsonl"

    names = label_names()
    rows = read_jsonl(args.input)
    flags, report = audit(rows, names)
    flags = merge_semantic_flags(
        rows,
        flags,
        args.semantic_dir,
        args.input,
    )

    print("\nOPT-009 QUALITY AUDIT")
    print("Candidates:", report["candidate_count"])
    print("Flagged:", report["flagged_candidates"])
    print("Flags:", report["flag_counts"])

    if args.verify:
        verify(rows, names, flags, args)

    counts = decisions(rows, flags)

    print("\nQUALITY STATUS")
    for status, count in sorted(counts.items()):
        print(f"{status:25} {count}")

    print("\nNo candidates approved for training.")
    print("Decisions:", DECISIONS)


if __name__ == "__main__":
    main()
