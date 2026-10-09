from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml.src.preprocessing.tokenizer import tokenize
from ml.src.preprocessing.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TmotionLab vocabulary from training data only."
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
        "--max-size",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--min-freq",
        type=int,
        default=2,
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

    vocab = Vocabulary(
        max_size=args.max_size,
        min_freq=args.min_freq,
    )

    token_sequences = (
        tokenize(
            text,
            lowercase=args.lowercase,
        )
        for text in frame["text"]
    )

    vocab.build(token_sequences)

    vocab.save(args.output)

    metadata_path = args.output.with_suffix(".metadata.json")

    metadata = {
        "source_split": "train",
        "source_file": str(args.train_file).replace("\\", "/"),
        "vocab_size": len(vocab),
        "max_size": args.max_size,
        "min_freq": args.min_freq,
        "lowercase": args.lowercase,
        "pad_token": "<PAD>",
        "pad_id": 0,
        "unk_token": "<UNK>",
        "unk_id": 1,
    }

    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Vocabulary size: {len(vocab)}")
    print(f"[OK] Vocabulary written: {args.output}")
    print(f"[OK] Metadata written: {metadata_path}")
    print("[OK] Vocabulary built from training split only")


if __name__ == "__main__":
    main()
