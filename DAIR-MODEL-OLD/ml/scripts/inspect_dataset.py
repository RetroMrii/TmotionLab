"""Inspect and validate a Hugging Face dataset against a YAML contract."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a dataset and validate its schema against a YAML config."
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to dataset YAML config.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON path for the inspection report.",
    )
    parser.add_argument(
        "--validate-config-only",
        action="store_true",
        help="Validate the YAML contract without importing/loading Hugging Face datasets.",
    )
    parser.add_argument(
        "--examples-per-split",
        type=int,
        default=3,
        help="Number of representative rows to include per split (default: 3).",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Dataset config must be a YAML mapping.")
    return config


def require_keys(config: dict[str, Any], keys: list[str]) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Dataset config missing required keys: {', '.join(missing)}")


def validate_contract(config: dict[str, Any], dataset: Any, class_label_type: type) -> dict[str, Any]:
    text_column = config["text_column"]
    label_column = config["label_column"]
    expected_labels = list(config["labels"])
    expected_sizes = config.get("expected_split_sizes", {})
    configured_splits = config["splits"]

    errors: list[str] = []
    report_splits: dict[str, Any] = {}

    expected_actual_splits = set(configured_splits.values())
    actual_splits = set(dataset.keys())
    missing_splits = sorted(expected_actual_splits - actual_splits)
    if missing_splits:
        errors.append(f"Missing configured splits: {missing_splits}")

    for logical_name, actual_name in configured_splits.items():
        if actual_name not in dataset:
            continue

        split = dataset[actual_name]
        columns = list(split.column_names)
        for column in (text_column, label_column):
            if column not in columns:
                errors.append(f"Split '{actual_name}' is missing column '{column}'.")

        size = len(split)
        expected_size = expected_sizes.get(logical_name)
        if expected_size is not None and size != expected_size:
            errors.append(
                f"Split '{actual_name}' has {size} rows; expected {expected_size}."
            )

        label_names: list[str] | None = None
        if label_column in split.features:
            label_feature = split.features[label_column]
            if isinstance(label_feature, class_label_type):
                label_names = list(label_feature.names)
                if label_names != expected_labels:
                    errors.append(
                        f"Label order mismatch in '{actual_name}': "
                        f"actual={label_names}, expected={expected_labels}."
                    )
            else:
                errors.append(
                    f"Column '{label_column}' in '{actual_name}' is not a ClassLabel feature."
                )

        texts = split[text_column] if text_column in columns else []
        labels = split[label_column] if label_column in columns else []
        empty_text_count = sum(
            1 for text in texts if text is None or not str(text).strip()
        )
        label_counts = Counter(labels)
        decoded_distribution = {
            expected_labels[label_id]: count
            for label_id, count in sorted(label_counts.items())
            if isinstance(label_id, int) and 0 <= label_id < len(expected_labels)
        }

        report_splits[logical_name] = {
            "actual_split_name": actual_name,
            "rows": size,
            "columns": columns,
            "label_names": label_names,
            "class_distribution": decoded_distribution,
            "empty_text_count": empty_text_count,
        }

    if errors:
        raise ValueError("Dataset contract validation failed:\n- " + "\n- ".join(errors))

    return {
        "dataset_name": config["dataset_name"],
        "hf_id": config["hf_id"],
        "hf_config": config.get("hf_config"),
        "revision": config.get("revision"),
        "task_type": config["task_type"],
        "text_column": text_column,
        "label_column": label_column,
        "labels": expected_labels,
        "splits": report_splits,
        "contract_valid": True,
    }


def add_examples(
    report: dict[str, Any],
    config: dict[str, Any],
    dataset: Any,
    examples_per_split: int,
) -> None:
    if examples_per_split < 0:
        raise ValueError("--examples-per-split must be >= 0.")

    text_column = config["text_column"]
    label_column = config["label_column"]
    labels = config["labels"]

    for logical_name, actual_name in config["splits"].items():
        if actual_name not in dataset:
            continue
        split = dataset[actual_name]
        examples = []
        for index in range(min(examples_per_split, len(split))):
            row = split[index]
            label_id = int(row[label_column])
            examples.append(
                {
                    "index": index,
                    "text": row[text_column],
                    "label_id": label_id,
                    "label": labels[label_id],
                }
            )
        report["splits"][logical_name]["examples"] = examples


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    require_keys(
        config,
        [
            "dataset_name",
            "hf_id",
            "task_type",
            "text_column",
            "label_column",
            "splits",
            "labels",
        ],
    )

    if args.validate_config_only:
        print(f"[OK] Config structure valid: {args.config}")
        return 0

    try:
        from datasets import ClassLabel, DatasetDict, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face 'datasets' is required for live inspection. "
            "Install project dependencies with: python -m pip install -r requirements.txt"
        ) from exc

    print(f"[INFO] Loading {config['hf_id']} (config={config.get('hf_config')!r})")
    dataset = load_dataset(
        config["hf_id"],
        config.get("hf_config"),
        revision=config.get("revision"),
    )
    if not isinstance(dataset, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(dataset).__name__}.")

    report = validate_contract(config, dataset, ClassLabel)
    add_examples(report, config, dataset, args.examples_per_split)

    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"[OK] Inspection report written to {args.output}")

    print("[OK] Dataset contract validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
