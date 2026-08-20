from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the DAIR.AI Emotion dataset for TmotionLab."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to dataset YAML config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for processed dataset output directory.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Dataset config must contain a YAML mapping.")

    return config


def stable_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_name = config["dataset_name"]
    hf_id = config["hf_id"]
    hf_config = config.get("hf_config")

    text_column = config["text_column"]
    label_column = config["label_column"]

    configured_splits = config["splits"]
    labels = config["labels"]

    label_id_to_name = {index: label_name for index, label_name in enumerate(labels)}

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path("data/processed") / dataset_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading dataset: {hf_id}")

    if hf_config:
        dataset = load_dataset(hf_id, hf_config)
    else:
        dataset = load_dataset(hf_id)

    metadata: dict = {
        "dataset_name": dataset_name,
        "hf_id": hf_id,
        "hf_config": hf_config,
        "text_column": text_column,
        "label_column": label_column,
        "labels": labels,
        "splits": {},
    }

    for logical_split, source_split in configured_splits.items():
        if source_split not in dataset:
            raise ValueError(
                f"Configured split '{source_split}' was not found in dataset."
            )

        split_dataset = dataset[source_split]

        rows = []

        for row_index, row in enumerate(split_dataset):
            text = row[text_column]
            label = int(row[label_column])

            if not isinstance(text, str):
                raise TypeError(
                    f"{source_split}[{row_index}] has non-string text: {type(text)}"
                )

            if label not in label_id_to_name:
                raise ValueError(
                    f"{source_split}[{row_index}] has unknown label id: {label}"
                )

            example_id = f"dair_{logical_split}_{row_index:06d}"

            rows.append(
                {
                    "example_id": example_id,
                    "text": text,
                    "label": label,
                    "label_name": label_id_to_name[label],
                    "text_sha256": stable_text_hash(text),
                }
            )

        frame = pd.DataFrame(rows)

        output_file = output_dir / f"{logical_split}.csv"

        frame.to_csv(
            output_file,
            index=False,
            encoding="utf-8",
        )

        metadata["splits"][logical_split] = {
            "source_split": source_split,
            "rows": len(frame),
            "output_file": str(output_file).replace("\\", "/"),
            "label_counts": {
                str(label_name): int(count)
                for label_name, count in (
                    frame["label_name"].value_counts().sort_index().items()
                )
            },
            "empty_text_count": int(frame["text"].fillna("").str.strip().eq("").sum()),
            "duplicate_text_count": int(frame["text"].duplicated().sum()),
        }

        print(f"[OK] {logical_split}: {len(frame)} rows -> {output_file}")

    metadata_path = output_dir / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Metadata written: {metadata_path}")
    print("[OK] Dataset preparation complete")


if __name__ == "__main__":
    main()
