from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.preprocessing.tokenizer import tokenize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze token-length distribution for TmotionLab."
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.train_file.exists():
        raise FileNotFoundError(f"Training file not found: {args.train_file}")

    frame = pd.read_csv(args.train_file)

    if "text" not in frame.columns:
        raise ValueError("Training file must contain a 'text' column")

    lengths = np.array(
        [len(tokenize(text, lowercase=args.lowercase)) for text in frame["text"]],
        dtype=np.int32,
    )

    if len(lengths) == 0:
        raise ValueError("Training file contains no rows")

    percentiles = {
        "p50": float(np.percentile(lengths, 50)),
        "p75": float(np.percentile(lengths, 75)),
        "p90": float(np.percentile(lengths, 90)),
        "p95": float(np.percentile(lengths, 95)),
        "p99": float(np.percentile(lengths, 99)),
    }

    summary = {
        "source_file": str(args.train_file).replace("\\", "/"),
        "num_examples": int(len(lengths)),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(lengths.mean()),
        "median_length": float(np.median(lengths)),
        "percentiles": percentiles,
        "counts_above": {
            "40": int((lengths > 40).sum()),
            "60": int((lengths > 60).sum()),
            "80": int((lengths > 80).sum()),
            "100": int((lengths > 100).sum()),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Examples analyzed: {len(lengths)}")
    print(f"[OK] Mean length: {summary['mean_length']:.2f}")
    print(f"[OK] Median length: {summary['median_length']:.2f}")
    print(f"[OK] P95 length: {percentiles['p95']:.2f}")
    print(f"[OK] P99 length: {percentiles['p99']:.2f}")
    print(f"[OK] Max length: {summary['max_length']}")
    print(f"[OK] Report written: {args.output}")


if __name__ == "__main__":
    main()
