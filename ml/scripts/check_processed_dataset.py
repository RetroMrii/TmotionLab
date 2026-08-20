from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "example_id",
    "text",
    "label",
    "label_name",
    "text_sha256",
}

EXPECTED_SPLITS = {
    "train": 16000,
    "validation": 2000,
    "test": 2000,
}

EXPECTED_LABELS = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate processed TmotionLab dataset files."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def validate_split(
    split_name: str,
    path: Path,
    expected_rows: int,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")

    frame = pd.read_csv(path)

    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{split_name}: missing columns: {sorted(missing_columns)}")

    if len(frame) != expected_rows:
        raise ValueError(
            f"{split_name}: expected {expected_rows} rows, found {len(frame)}"
        )

    if frame["example_id"].duplicated().any():
        raise ValueError(f"{split_name}: duplicate example_id values found")

    if frame["example_id"].isna().any():
        raise ValueError(f"{split_name}: null example_id values found")

    if frame["text"].isna().any():
        raise ValueError(f"{split_name}: null text values found")

    if frame["text"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{split_name}: empty text values found")

    unknown_labels = set(frame["label"].astype(int).unique()) - set(EXPECTED_LABELS)

    if unknown_labels:
        raise ValueError(f"{split_name}: unknown label IDs: {sorted(unknown_labels)}")

    for label_id, label_name in EXPECTED_LABELS.items():
        rows = frame.loc[frame["label"].astype(int) == label_id]

        if not rows.empty and not rows["label_name"].eq(label_name).all():
            raise ValueError(
                f"{split_name}: label mapping mismatch for {label_id} -> {label_name}"
            )

    print(f"[OK] {split_name}: {len(frame)} rows validated")

    return frame


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir

    frames = {}

    for split_name, expected_rows in EXPECTED_SPLITS.items():
        frames[split_name] = validate_split(
            split_name,
            dataset_dir / f"{split_name}.csv",
            expected_rows,
        )

    metadata_path = dataset_dir / "metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    all_ids = pd.concat(
        [frame[["example_id"]] for frame in frames.values()],
        ignore_index=True,
    )

    if all_ids["example_id"].duplicated().any():
        raise ValueError("example_id collision detected across dataset splits")

    print("[OK] example_id values are unique across all splits")
    print(f"[OK] Metadata loaded: {metadata_path}")
    print("[OK] Processed dataset validation complete")


if __name__ == "__main__":
    main()
