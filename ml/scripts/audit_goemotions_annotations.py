import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
SIMPLIFIED_DIR = RAW_DIR / "goemotions_simplified_v001"
ANNOTATIONS_DIR = RAW_DIR / "goemotions_raw_v001"

OUTPUT_DIR = ROOT / "reports" / "goemotions"
OUTPUT_PATH = OUTPUT_DIR / "annotation_audit_v001.json"

SPLITS = ("train", "validation", "test")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON: {path}:{line_number}") from exc


def verify_hash(path, expected):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    if digest.hexdigest() != expected:
        raise ValueError(f"Checksum mismatch: {path}")


def normalize_text(text):
    return " ".join(text.casefold().split())


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {OUTPUT_PATH}")

    simplified_manifest = json.loads(
        (SIMPLIFIED_DIR / "manifest.json").read_text(encoding="utf-8")
    )
    raw_manifest = json.loads(
        (ANNOTATIONS_DIR / "manifest.json").read_text(encoding="utf-8")
    )

    if simplified_manifest["source_revision"] != raw_manifest["source_revision"]:
        raise ValueError("Dataset revisions do not match")

    labels = simplified_manifest["label_names"]

    if labels != raw_manifest["label_names"]:
        raise ValueError("Label mappings do not match")

    if len(labels) != 28:
        raise ValueError("Expected 28 labels")

    # Load official benchmark identities.
    split_ids = {}
    split_texts = {}
    simplified_rows = {}
    all_source_ids = set()

    for split in SPLITS:
        info = simplified_manifest["splits"][split]
        path = SIMPLIFIED_DIR / info["file"]
        verify_hash(path, info["sha256"])

        ids = set()
        texts = set()
        rows = 0

        for row in read_jsonl(path):
            source_id = row["id"]
            text = normalize_text(row["text"])

            if not isinstance(source_id, str) or not source_id:
                raise ValueError("Invalid simplified source ID")

            if source_id in ids:
                raise ValueError(f"Duplicate simplified ID: {source_id}")

            if source_id in all_source_ids:
                raise ValueError(f"Cross-split source ID: {source_id}")

            ids.add(source_id)
            all_source_ids.add(source_id)
            texts.add(text)
            rows += 1

        if rows != info["row_count"]:
            raise ValueError(f"Row count mismatch: {split}")

        split_ids[split] = ids
        split_texts[split] = texts
        simplified_rows[split] = rows

    # Audit exact normalized-text overlap between splits.
    split_text_overlap = {}

    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1 :]:
            key = f"{left}_vs_{right}"
            split_text_overlap[key] = len(split_texts[left] & split_texts[right])

    # Aggregate annotation votes per comment.
    annotations_path = ANNOTATIONS_DIR / raw_manifest["file"]

    verify_hash(
        annotations_path,
        raw_manifest["sha256"],
    )

    comment_stats = {}
    annotation_count = 0
    unclear_count = 0
    label_vote_counts = Counter()

    for row in read_jsonl(annotations_path):
        comment_id = row["id"]
        text = normalize_text(row["text"])

        if not isinstance(comment_id, str) or not comment_id:
            raise ValueError("Invalid raw comment ID")

        if comment_id not in comment_stats:
            comment_stats[comment_id] = {
                "text": text,
                "n": 0,
                "votes": [0] * len(labels),
            }

        stats = comment_stats[comment_id]

        if stats["text"] != text:
            raise ValueError(f"Inconsistent text for comment: {comment_id}")

        stats["n"] += 1
        annotation_count += 1

        if row["example_very_unclear"]:
            unclear_count += 1

        for index, label in enumerate(labels):
            vote = row[label]

            if type(vote) not in (bool, int) or vote not in (0, 1):
                raise ValueError(f"Invalid vote for {label}: {vote}")

            stats["votes"][index] += int(vote)

            if vote:
                label_vote_counts[label] += 1

    if annotation_count != raw_manifest["row_count"]:
        raise ValueError("Raw annotation count mismatch")

    if len(comment_stats) != raw_manifest["unique_comment_count"]:
        raise ValueError("Unique comment count mismatch")

    if unclear_count != raw_manifest["unclear_annotation_count"]:
        raise ValueError("Unclear annotation count mismatch")

    # Analyze annotator voting patterns.
    annotations_per_comment = Counter()
    unanimous_comments = 0
    disputed_comments = 0

    for stats in comment_stats.values():
        n = stats["n"]
        annotations_per_comment[n] += 1

        if n < 2:
            continue

        unanimous = all(votes == 0 or votes == n for votes in stats["votes"])

        if unanimous:
            unanimous_comments += 1
        else:
            disputed_comments += 1

    # Audit source-ID overlap with official splits.
    raw_ids = set(comment_stats)

    source_id_overlap = {split: len(raw_ids & split_ids[split]) for split in SPLITS}

    # Candidate inventory only: NOT approved training data.
    heldout_ids = split_ids["validation"] | split_ids["test"]
    heldout_texts = split_texts["validation"] | split_texts["test"]

    candidate_count = 0
    excluded_by_id = 0
    excluded_by_text = 0

    for comment_id, stats in comment_stats.items():
        if comment_id in heldout_ids:
            excluded_by_id += 1
        elif stats["text"] in heldout_texts:
            excluded_by_text += 1
        else:
            candidate_count += 1

    result = {
        "audit_id": "goemotions_annotation_audit_v001",
        "source_revision": raw_manifest["source_revision"],
        "raw_annotation_rows": annotation_count,
        "unique_raw_comments": len(comment_stats),
        "unclear_annotation_rows": unclear_count,
        "simplified_split_counts": simplified_rows,
        "annotations_per_comment": dict(sorted(annotations_per_comment.items())),
        "unanimous_multi_annotator_comments": unanimous_comments,
        "disputed_multi_annotator_comments": disputed_comments,
        "label_vote_counts": {label: label_vote_counts[label] for label in labels},
        "raw_source_id_overlap": source_id_overlap,
        "simplified_cross_split_text_overlap": split_text_overlap,
        "raw_candidate_inventory": {
            "excluded_by_heldout_id": excluded_by_id,
            "excluded_by_heldout_normalized_text": excluded_by_text,
            "remaining_comment_ids": candidate_count,
            "approved_for_training": False,
        },
        "limitations": [
            "Raw candidate counts do not imply additional independent training samples.",
            "Unanimous and disputed counts are descriptive, not chance-corrected agreement.",
            "Unclear annotations remain included in raw vote statistics.",
            "Source-ID and normalized exact-text checks do not detect near duplicates.",
            "No alternative targets or synthetic examples are approved by this audit.",
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Raw annotation rows: {annotation_count}")
    print(f"Unique raw comments: {len(comment_stats)}")
    print(f"Unclear annotation rows: {unclear_count}")
    print(f"Unanimous comments (2+ raters): {unanimous_comments}")
    print(f"Disputed comments (2+ raters): {disputed_comments}")

    print("\nRaw source-ID overlap:")
    for split, count in source_id_overlap.items():
        print(f"  {split}: {count}")

    print("\nSimplified normalized-text overlap:")
    for pair, count in split_text_overlap.items():
        print(f"  {pair}: {count}")

    print("\nRaw candidate inventory:")
    print(f"  Held-out ID exclusions: {excluded_by_id}")
    print(f"  Held-out text exclusions: {excluded_by_text}")
    print(f"  Remaining comments: {candidate_count}")

    print(f"\n[OK] Audit saved: {OUTPUT_PATH}")
    print("[OK] P11-B4 annotation audit completed")


if __name__ == "__main__":
    main()
