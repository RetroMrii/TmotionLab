import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "google-research-datasets/go_emotions"
CONFIG = "raw"
REVISION = "add492243ff905527e67aeb8b80c082af02207c3"

DATASET_VERSION = "goemotions_raw_v001"
EXPECTED_ROWS = 211225

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_VERSION
SIMPLIFIED_MANIFEST = (
    PROJECT_ROOT / "data" / "raw" / "goemotions_simplified_v001" / "manifest.json"
)

METADATA_FIELDS = [
    "text",
    "id",
    "author",
    "subreddit",
    "link_id",
    "parent_id",
    "created_utc",
    "rater_id",
    "example_very_unclear",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    if not SIMPLIFIED_MANIFEST.exists():
        raise FileNotFoundError(
            "Simplified dataset manifest not found. Complete P11-B2 first."
        )

    simplified = json.loads(SIMPLIFIED_MANIFEST.read_text(encoding="utf-8"))

    if (
        simplified["source_dataset"] != DATASET_ID
        or simplified["source_revision"] != REVISION
        or simplified["source_config"] != "simplified"
    ):
        raise ValueError("Raw and simplified source revisions do not match")

    label_names = simplified["label_names"]

    if len(label_names) != 28:
        raise ValueError("Expected 28 emotion labels")

    required_fields = METADATA_FIELDS + label_names

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / "annotations.jsonl"
    manifest_path = OUTPUT_DIR / "manifest.json"

    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "Raw annotation snapshot already exists or is "
            "incomplete. Inspect it before rerunning ingestion."
        )

    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        revision=REVISION,
    )

    if set(dataset.keys()) != {"train"}:
        raise ValueError(f"Unexpected raw dataset splits: {list(dataset.keys())}")

    rows = dataset["train"]

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, got {len(rows)}")

    missing = set(required_fields) - set(rows.column_names)

    if missing:
        raise ValueError(f"Missing raw fields: {sorted(missing)}")

    print(f"Dataset: {DATASET_ID}")
    print(f"Config: {CONFIG}")
    print(f"Revision: {REVISION}")
    print(f"Annotation rows: {len(rows)}")

    temp_path = OUTPUT_DIR / "annotations.jsonl.tmp"
    unique_comment_ids = set()
    unclear_count = 0

    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            for index, row in enumerate(rows):
                record = {field: row[field] for field in required_fields}

                comment_id = record["id"]

                if not isinstance(comment_id, str) or not comment_id:
                    raise ValueError(f"Invalid comment ID at row {index}")

                unique_comment_ids.add(comment_id)

                for label in label_names:
                    if record[label] not in (0, 1):
                        raise ValueError(f"Invalid binary label {label} at row {index}")

                if not isinstance(record["example_very_unclear"], bool):
                    raise ValueError(f"Invalid unclear flag at row {index}")

                if record["example_very_unclear"]:
                    unclear_count += 1

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )

        temp_path.replace(output_path)

    finally:
        temp_path.unlink(missing_ok=True)

    manifest = {
        "dataset_version": DATASET_VERSION,
        "source_dataset": DATASET_ID,
        "source_config": CONFIG,
        "source_revision": REVISION,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_semantics": "individual_annotator_judgment",
        "row_count": len(rows),
        "unique_comment_count": len(unique_comment_ids),
        "unclear_annotation_count": unclear_count,
        "label_count": len(label_names),
        "label_names": label_names,
        "fields": required_fields,
        "file": output_path.name,
        "sha256": sha256_file(output_path),
        "simplified_dataset_version": (simplified["dataset_version"]),
    }

    manifest_path.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Unique comment IDs: {len(unique_comment_ids)}")
    print(f"Unclear annotations: {unclear_count}")
    print(f"[OK] Annotations: {output_path}")
    print(f"[OK] Manifest: {manifest_path}")
    print("[OK] GoEmotions raw ingestion completed")


if __name__ == "__main__":
    main()
