import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset


DATASET_ID = "google-research-datasets/go_emotions"
CONFIG = "simplified"
REVISION = "add492243ff905527e67aeb8b80c082af02207c3"

DATASET_VERSION = "goemotions_simplified_v001"

EXPECTED_SPLITS = {
    "train": 43410,
    "validation": 5426,
    "test": 5427,
}

EXPECTED_LABEL_COUNT = 28

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / DATASET_VERSION


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_jsonl(path: Path, rows) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing raw snapshot: {path}")

    temp_path = path.with_suffix(path.suffix + ".tmp")

    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as file:
            for row in rows:
                record = {
                    "id": row["id"],
                    "text": row["text"],
                    "labels": row["labels"],
                }

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )

        temp_path.replace(path)

    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_path = OUTPUT_DIR / "manifest.json"

    expected_paths = {split: OUTPUT_DIR / f"{split}.jsonl" for split in EXPECTED_SPLITS}

    existing_paths = [
        path for path in [manifest_path, *expected_paths.values()] if path.exists()
    ]

    if existing_paths:
        raise FileExistsError(
            "Raw snapshot already exists or is incomplete. "
            "Do not overwrite it. Inspect these files first: "
            + ", ".join(str(path) for path in existing_paths)
        )

    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        revision=REVISION,
    )

    if set(dataset.keys()) != set(EXPECTED_SPLITS):
        raise ValueError("Unexpected dataset splits")

    label_names = dataset["train"].features["labels"].feature.names

    if len(label_names) != EXPECTED_LABEL_COUNT:
        raise ValueError("Unexpected label count")

    if label_names[-1] != "neutral":
        raise ValueError("Unexpected label mapping")

    manifest = {
        "dataset_version": DATASET_VERSION,
        "source_dataset": DATASET_ID,
        "source_config": CONFIG,
        "source_revision": REVISION,
        "ingested_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "multi_label_classification",
        "label_count": EXPECTED_LABEL_COUNT,
        "label_names": label_names,
        "source_fields": ["id", "text", "labels"],
        "splits": {},
    }

    for split_name, expected_count in EXPECTED_SPLITS.items():
        split = dataset[split_name]

        if len(split) != expected_count:
            raise ValueError(
                f"{split_name}: expected {expected_count}, got {len(split)}"
            )

        if split.features["labels"].feature.names != label_names:
            raise ValueError("Inconsistent label mapping across splits")

        if not {"id", "text", "labels"}.issubset(split.column_names):
            raise ValueError("Unexpected source schema")

    for split_name in EXPECTED_SPLITS:
        split = dataset[split_name]
        output_path = expected_paths[split_name]

        save_jsonl(output_path, split)

        manifest["splits"][split_name] = {
            "row_count": len(split),
            "file": output_path.name,
            "sha256": sha256_file(output_path),
            "multi_label_count": sum(len(row["labels"]) > 1 for row in split),
        }

        print(f"[OK] {split_name}: {len(split)} rows -> {output_path}")

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"[OK] Manifest: {manifest_path}")
    print("[OK] GoEmotions raw ingestion completed")


if __name__ == "__main__":
    main()
