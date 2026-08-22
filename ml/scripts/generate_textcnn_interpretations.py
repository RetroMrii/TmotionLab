from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from ml.src.interpretation.textcnn_interpreter import (
    interpret_textcnn_input,
    load_textcnn_checkpoint,
)
from ml.src.preprocessing.sequence import encode_text
from ml.src.preprocessing.tokenizer import tokenize
from ml.src.preprocessing.vocab import Vocabulary


LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible TextCNN interpretation "
            "artifacts for held-out examples."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/configs/textcnn_dair_v001.yaml"),
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("ml/artifacts/predictions/textcnn_dair_v001_test_predictions.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "ml/artifacts/interpretations/textcnn_dair_v001_interpretations.jsonl"
        ),
    )

    parser.add_argument(
        "--examples-per-cohort",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def select_examples(
    frame: pd.DataFrame,
    *,
    examples_per_cohort: int,
) -> pd.DataFrame:
    correct = frame[frame["true_label"] == frame["predicted_label"]].copy()

    wrong = frame[frame["true_label"] != frame["predicted_label"]].copy()

    correct_high = (
        correct.sort_values(
            "top_score",
            ascending=False,
        )
        .head(examples_per_cohort)
        .assign(interpretation_cohort="correct_high_score")
    )

    correct_low = (
        correct.sort_values(
            "top_score",
            ascending=True,
        )
        .head(examples_per_cohort)
        .assign(interpretation_cohort="correct_low_score")
    )

    wrong_high = (
        wrong.sort_values(
            "top_score",
            ascending=False,
        )
        .head(examples_per_cohort)
        .assign(interpretation_cohort="wrong_high_score")
    )

    wrong_low = (
        wrong.sort_values(
            "top_score",
            ascending=True,
        )
        .head(examples_per_cohort)
        .assign(interpretation_cohort="wrong_low_score")
    )

    return pd.concat(
        [
            correct_high,
            correct_low,
            wrong_high,
            wrong_low,
        ],
        ignore_index=True,
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    if not args.predictions.exists():
        raise FileNotFoundError(f"Predictions not found: {args.predictions}")

    predictions = pd.read_csv(args.predictions)

    selected = select_examples(
        predictions,
        examples_per_cohort=(args.examples_per_cohort),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing = config["preprocessing"]

    vocab = Vocabulary.load(preprocessing["vocab_path"])

    model, checkpoint = load_textcnn_checkpoint(
        config["artifacts"]["model"],
        device=device,
    )

    max_length = int(preprocessing["max_sequence_length"])

    kernel_sizes = [int(value) for value in checkpoint["kernel_sizes"]]

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    padding_counts = []

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for _, row in selected.iterrows():
            text = str(row["text"])

            tokens = tokenize(
                text,
                lowercase=bool(preprocessing["lowercase"]),
            )[:max_length]

            input_ids = encode_text(
                tokens,
                vocab,
                max_length=max_length,
            )

            tensor = torch.tensor(
                [input_ids],
                dtype=torch.long,
                device=device,
            )

            interpretation = interpret_textcnn_input(
                model,
                input_ids=tensor,
                tokens=tokens,
                kernel_sizes=kernel_sizes,
                labels=LABELS,
                top_k=args.top_k,
            )

            padding_counts.append(interpretation["padding_activation_count"])

            record = {
                "model_id": config["experiment_id"],
                "example_id": str(row["example_id"]),
                "cohort": row["interpretation_cohort"],
                "text": text,
                "true_label": str(row["true_label_name"]),
                "saved_prediction": str(row["predicted_label_name"]),
                "saved_top_score": float(row["top_score"]),
                "tokens": tokens,
                "interpretation": (interpretation),
            }

            json.dump(
                record,
                handle,
                ensure_ascii=False,
            )

            handle.write("\n")

    print(f"[OK] Interpretation examples: {len(selected)}")

    print(
        "[RESULT] Mean padding-max filters: "
        f"{sum(padding_counts) / len(padding_counts):.2f}"
    )

    print(f"[RESULT] Max padding-max filters: {max(padding_counts)}")

    print(f"[OK] Interpretation artifact: {args.output}")


if __name__ == "__main__":
    main()
