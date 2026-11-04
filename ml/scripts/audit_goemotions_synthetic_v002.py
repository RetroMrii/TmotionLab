import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.src.data.goemotions_dataset import (
    load_goemotions_rows,
)
from ml.scripts.generate_goemotions_synthetic_v002 import (
    BATCH_DIR,
    CANDIDATES_PATH,
    MANIFEST_PATH,
    build_jobs,
    load_json,
    load_spec,
    manifest_for,
    normalize,
    verify_existing,
)


ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = ROOT / "reports/goemotions/synthetic_pilot_v002_audit.json"

REVIEW_PATH = ROOT / "data/synthetic/goemotions_pilot_v002/review_queue.jsonl"

NEAR_DUPLICATE_THRESHOLD = 0.82

SUSPICIOUS_STRINGS = (
    "\ufffd",
    "â€™",
    "â€œ",
    "â€",
    "ג€™",
)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def find_near_duplicates(candidates):
    texts = [normalize(item["text"]) for item in candidates]

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=False,
        norm="l2",
    )

    features = vectorizer.fit_transform(texts)
    similarities = (features @ features.T).tocoo()

    pairs = []

    for i, j, value in zip(
        similarities.row,
        similarities.col,
        similarities.data,
    ):
        if i >= j or value < NEAR_DUPLICATE_THRESHOLD:
            continue

        pairs.append(
            {
                "candidate_a": candidates[i]["candidate_id"],
                "candidate_b": candidates[j]["candidate_id"],
                "similarity": round(float(value), 4),
            }
        )

    pairs.sort(key=lambda pair: -pair["similarity"])

    return pairs


def main():
    if REPORT_PATH.exists() or REVIEW_PATH.exists():
        raise FileExistsError(
            "Audit output already exists. Preserve the original audit version."
        )

    config, _ = load_spec()
    jobs = build_jobs(config)
    manifest = manifest_for(config, jobs)

    if load_json(MANIFEST_PATH) != manifest:
        raise ValueError("Generation manifest mismatch")

    candidates = read_jsonl(CANDIDATES_PATH)

    if len(candidates) != 300:
        raise ValueError(f"Expected 300 candidates, found {len(candidates)}")

    # Validate the original batch records against the
    # frozen specification and derived candidate file.
    expected_candidates = []
    response_ids = set()
    returned_models = Counter()

    for job in jobs:
        path = BATCH_DIR / f"{job['batch_id']}.json"

        if not path.is_file():
            raise FileNotFoundError(path)

        record = load_json(path)
        parsed = verify_existing(record, job, manifest)

        if not record.get("response_id"):
            raise ValueError("Missing API response ID")

        if record["response_id"] in response_ids:
            raise ValueError("Duplicate API response ID")

        response_ids.add(record["response_id"])
        returned_models[record["returned_model"]] += 1

        for slot, example in enumerate(parsed.examples):
            expected_candidates.append(
                {
                    "candidate_id": (f"syn_v002_{job['batch_id']}_{slot:02d}"),
                    "text": example.text.strip(),
                    "primary": job["label"],
                    "secondary": example.secondary_labels,
                    "scenario": job["scenarios"][slot],
                    "batch_id": job["batch_id"],
                    "response_id": record["response_id"],
                }
            )

    label_counts = Counter()
    scenario_counts = defaultdict(Counter)
    normalized_groups = defaultdict(list)
    flags_by_id = defaultdict(list)

    for actual, expected in zip(
        candidates,
        expected_candidates,
    ):
        cid = expected["candidate_id"]

        if (
            actual["candidate_id"] != cid
            or actual["text"] != expected["text"]
            or actual["intended_primary_label"] != expected["primary"]
            or actual["proposed_secondary_labels"] != expected["secondary"]
            or actual["assigned_scenario"] != expected["scenario"]
            or actual["batch_id"] != expected["batch_id"]
            or actual["response_id"] != expected["response_id"]
            or actual["approval_status"] != "pending"
            or actual["label_status"] != "unverified"
            or actual["config_sha256"] != manifest["config_sha256"]
            or actual["prompt_sha256"] != manifest["prompt_sha256"]
        ):
            raise ValueError(f"Candidate provenance mismatch: {cid}")

        label_counts[expected["primary"]] += 1
        scenario_counts[expected["primary"]][expected["scenario"]] += 1

        normalized_groups[normalize(actual["text"])].append(cid)

        if any(marker in actual["text"] for marker in SUSPICIOUS_STRINGS):
            flags_by_id[cid].append("suspected_encoding_corruption")

    if dict(label_counts) != config["targets"]:
        raise ValueError("Incorrect target-label counts")

    # Exact and normalized comparisons only.
    # Held-out texts are not used as generation prompts.
    overlap_by_split = {}
    overlapping_ids = set()

    for split in ("train", "validation", "test"):
        source_rows, _ = load_goemotions_rows(split)

        source_texts = {normalize(row["text"]) for row in source_rows}

        matching_ids = [
            item["candidate_id"]
            for item in candidates
            if normalize(item["text"]) in source_texts
        ]

        overlap_by_split[split] = matching_ids
        overlapping_ids.update(matching_ids)

        for cid in matching_ids:
            flags_by_id[cid].append(f"normalized_overlap_{split}")

    duplicate_groups = [ids for ids in normalized_groups.values() if len(ids) > 1]

    for group in duplicate_groups:
        for cid in group:
            flags_by_id[cid].append("duplicate_synthetic_text")

    near_duplicate_pairs = find_near_duplicates(candidates)

    for pair in near_duplicate_pairs:
        for key in ("candidate_a", "candidate_b"):
            flags_by_id[pair[key]].append("near_duplicate_candidate")

    review_rows = []

    for item in candidates:
        cid = item["candidate_id"]
        flags = sorted(set(flags_by_id[cid]))

        review_rows.append(
            {
                "candidate_id": cid,
                "text": item["text"],
                "intended_primary_label": (item["intended_primary_label"]),
                "proposed_secondary_labels": (item["proposed_secondary_labels"]),
                "assigned_scenario": (item["assigned_scenario"]),
                "automated_flags": flags,
                "review_status": "pending",
                "approved_labels": None,
                "review_notes": "",
            }
        )

    report = {
        "audit_id": "synthetic_pilot_v002_audit",
        "experiment_id": config["experiment_id"],
        "candidate_count": len(candidates),
        "batch_count": len(jobs),
        "candidate_file_sha256": sha256_file(CANDIDATES_PATH),
        "generation_manifest_sha256": sha256_file(MANIFEST_PATH),
        "primary_label_counts": dict(label_counts),
        "scenario_counts": {
            label: dict(counts) for label, counts in scenario_counts.items()
        },
        "returned_models": dict(returned_models),
        "normalized_duplicate_groups": (duplicate_groups),
        "normalized_overlap_by_split": (overlap_by_split),
        "near_duplicate_threshold": (NEAR_DUPLICATE_THRESHOLD),
        "near_duplicate_pairs": (near_duplicate_pairs),
        "flagged_candidate_count": sum(
            bool(row["automated_flags"]) for row in review_rows
        ),
        "automatic_approval_count": 0,
        "semantic_review_completed": False,
        "training_approved": False,
        "limitations": [
            "No semantic label validation performed.",
            "Near-duplicate similarity is lexical, not semantic equivalence.",
            "Real-data overlap checks normalize case and whitespace only.",
            "Validation and test text accessed solely for exclusion checks.",
        ],
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(
            report,
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")

    with REVIEW_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        for row in review_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("P11-E2D — SYNTHETIC DATA AUDIT")
    print("-" * 44)
    print("Candidates:", len(candidates))
    print("Batches:", len(jobs))

    for label, count in label_counts.items():
        print(f"  {label:<13} {count}")

    print(
        "Normalized duplicate groups:",
        len(duplicate_groups),
    )

    for split, ids in overlap_by_split.items():
        print(f"Overlap with {split}: {len(ids)}")

    print(
        "Near-duplicate pairs:",
        len(near_duplicate_pairs),
    )
    print(
        "Flagged candidates:",
        report["flagged_candidate_count"],
    )
    print("Automatically approved: 0")
    print(f"[OK] Audit saved: {REPORT_PATH}")
    print(f"[OK] Review queue saved: {REVIEW_PATH}")


if __name__ == "__main__":
    main()
