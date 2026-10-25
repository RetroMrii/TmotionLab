import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/goemotions_simplified_v001"
DEST = ROOT / "data/processed/goemotions_v001"

SPLITS = ("train", "validation", "test")


def normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def write_jsonl(path: Path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )


def main():
    source_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))

    source_rows = {}

    for split in SPLITS:
        info = source_manifest["splits"][split]
        path = SOURCE / info["file"]

        if sha256(path) != info["sha256"]:
            raise ValueError(f"Source checksum mismatch: {split}")

        rows = list(read_rows(path))

        if len(rows) != info["row_count"]:
            raise ValueError(f"Source count mismatch: {split}")

        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate source IDs: {split}")

        source_rows[split] = rows

    label_names = source_manifest["label_names"]

    if len(label_names) != 28:
        raise ValueError("Unexpected label taxonomy")

    all_ids = set()

    for split, rows in source_rows.items():
        for row in rows:
            if set(row) != {"id", "text", "labels"}:
                raise ValueError(f"Unexpected row fields: {split}")

            if row["id"] in all_ids:
                raise ValueError("Cross-split source ID overlap")

            all_ids.add(row["id"])

            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("Invalid text")

            labels = row["labels"]

            if (
                not isinstance(labels, list)
                or not labels
                or len(set(labels)) != len(labels)
                or any(
                    type(value) is not int or not 0 <= value < 28 for value in labels
                )
            ):
                raise ValueError("Invalid multi-label target")

    heldout_texts = {
        normalize(row["text"])
        for split in ("validation", "test")
        for row in source_rows[split]
    }

    kept = []
    excluded = []

    for index, row in enumerate(source_rows["train"]):
        record = {
            "example_id": f"goe_train_{index:06d}",
            "source_id": row["id"],
            "text": row["text"],
            "labels": row["labels"],
            "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
        }

        if normalize(row["text"]) in heldout_texts:
            excluded.append(
                {
                    "example_id": record["example_id"],
                    "source_id": record["source_id"],
                    "reason": "heldout_normalized_text_overlap",
                }
            )
        else:
            kept.append(record)

    # Validation and test retain published row order.
    processed = {"train": kept}

    for split in ("validation", "test"):
        processed[split] = [
            {
                "example_id": f"goe_{split}_{index:06d}",
                "source_id": row["id"],
                "text": row["text"],
                "labels": row["labels"],
                "text_sha256": hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
            }
            for index, row in enumerate(source_rows[split])
        ]

    if any(normalize(row["text"]) in heldout_texts for row in kept):
        raise AssertionError("Training leakage check failed")

    DEST.mkdir(parents=True, exist_ok=True)

    paths = {split: DEST / f"{split}.jsonl" for split in SPLITS}
    excluded_path = DEST / "excluded_train.jsonl"
    manifest_path = DEST / "manifest.json"

    if any(
        path.exists()
        for path in [
            *paths.values(),
            excluded_path,
            manifest_path,
        ]
    ):
        raise FileExistsError(
            "Processed outputs already exist. Do not overwrite this dataset version."
        )

    for split in SPLITS:
        write_jsonl(paths[split], processed[split])

    write_jsonl(excluded_path, excluded)

    manifest = {
        "dataset_version": "goemotions_v001",
        "source_version": source_manifest["dataset_version"],
        "source_revision": source_manifest["source_revision"],
        "label_names": label_names,
        "policy": "exclude train/heldout normalized-text overlap",
        "normalization": "casefold + whitespace collapse",
        "source_train_rows": len(source_rows["train"]),
        "excluded_train_rows": len(excluded),
        "processed_splits": {
            split: {
                "rows": len(processed[split]),
                "sha256": sha256(paths[split]),
            }
            for split in SPLITS
        },
        "excluded_file_sha256": sha256(excluded_path),
        "validation_test_overlap_preserved": True,
        "note": (
            "Official validation and test membership preserved. "
            "Their mutual text overlap remains an evaluation caveat."
        ),
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Source train rows: {len(source_rows['train'])}")
    print(f"Excluded train rows: {len(excluded)}")

    for split in SPLITS:
        print(f"{split}: {len(processed[split])} rows")

    print(f"[OK] Manifest: {manifest_path}")
    print("[OK] P11-B5 processed splits created")


if __name__ == "__main__":
    main()
