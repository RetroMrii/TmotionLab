from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Aggregate TextCNN interpretation artifacts for TmotionLab.")
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "ml/artifacts/interpretations/textcnn_dair_v001_interpretations.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/interpretations"),
    )

    parser.add_argument(
        "--top-spans-per-example",
        type=int,
        default=10,
    )

    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Interpretation artifact not found: {path}")

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number}") from exc

            records.append(record)

    if not records:
        raise ValueError("Interpretation artifact contains no records")

    return records


def validate_record(record: dict) -> None:
    required = {
        "model_id",
        "example_id",
        "cohort",
        "text",
        "true_label",
        "saved_prediction",
        "saved_top_score",
        "tokens",
        "interpretation",
    }

    missing = required - set(record)

    if missing:
        raise ValueError(f"Interpretation record missing keys: {sorted(missing)}")

    interpretation = record["interpretation"]

    required_interpretation = {
        "prediction",
        "scores",
        "top_activations",
        "top_text_spans",
        "padding_activation_count",
    }

    missing_interpretation = required_interpretation - set(interpretation)

    if missing_interpretation:
        raise ValueError(
            f"Interpretation payload missing keys: {sorted(missing_interpretation)}"
        )

    if interpretation["prediction"]["label"] != record["saved_prediction"]:
        raise ValueError(
            "Saved prediction does not match "
            "interpretation prediction for "
            f"{record['example_id']}"
        )


def build_filter_rows(
    records: list[dict],
    *,
    top_spans_per_example: int,
) -> list[dict]:
    rows = []

    for record in records:
        spans = record["interpretation"]["top_text_spans"][:top_spans_per_example]

        for rank, span in enumerate(
            spans,
            start=1,
        ):
            rows.append(
                {
                    "example_id": (record["example_id"]),
                    "cohort": record["cohort"],
                    "true_label": (record["true_label"]),
                    "predicted_label": (record["saved_prediction"]),
                    "correct": (record["true_label"] == record["saved_prediction"]),
                    "rank": rank,
                    "kernel_size": int(span["kernel_size"]),
                    "filter_index": int(span["filter_index"]),
                    "activation": float(span["activation"]),
                    "span_text": str(span["span_text"]),
                    "token_start": int(span["token_start"]),
                    "token_end_exclusive": int(span["token_end_exclusive"]),
                }
            )

    return rows


def build_filter_frequency(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    grouped = (
        rows.groupby(
            [
                "kernel_size",
                "filter_index",
                "predicted_label",
            ]
        )
        .agg(
            occurrences=(
                "example_id",
                "count",
            ),
            unique_examples=(
                "example_id",
                "nunique",
            ),
            mean_activation=(
                "activation",
                "mean",
            ),
            mean_rank=(
                "rank",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "unique_examples",
                "occurrences",
                "mean_activation",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
    )

    return grouped


def build_span_frequency(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    cleaned = rows.copy()

    cleaned["span_text_normalized"] = (
        cleaned["span_text"].astype(str).str.strip().str.lower()
    )

    grouped = (
        cleaned.groupby(
            [
                "span_text_normalized",
                "predicted_label",
            ]
        )
        .agg(
            occurrences=(
                "example_id",
                "count",
            ),
            unique_examples=(
                "example_id",
                "nunique",
            ),
            mean_activation=(
                "activation",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            [
                "unique_examples",
                "occurrences",
                "mean_activation",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
    )

    return grouped


def build_padding_summary(
    records: list[dict],
) -> pd.DataFrame:
    rows = []

    for record in records:
        padding_count = int(record["interpretation"]["padding_activation_count"])

        rows.append(
            {
                "example_id": (record["example_id"]),
                "cohort": record["cohort"],
                "true_label": (record["true_label"]),
                "predicted_label": (record["saved_prediction"]),
                "correct": (record["true_label"] == record["saved_prediction"]),
                "token_count": len(record["tokens"]),
                "padding_activation_count": (padding_count),
            }
        )

    return pd.DataFrame(rows)


def top_records(
    frame: pd.DataFrame,
    *,
    n: int = 20,
) -> list[dict]:
    if frame.empty:
        return []

    return frame.head(n).to_dict(orient="records")


def main() -> None:
    args = parse_args()

    records = load_jsonl(args.input)

    for record in records:
        validate_record(record)

    rows = pd.DataFrame(
        build_filter_rows(
            records,
            top_spans_per_example=(args.top_spans_per_example),
        )
    )

    filter_frequency = build_filter_frequency(rows)

    span_frequency = build_span_frequency(rows)

    padding_summary = build_padding_summary(records)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filter_path = args.output_dir / ("textcnn_dair_v001_filter_frequency.csv")

    span_path = args.output_dir / ("textcnn_dair_v001_span_frequency.csv")

    padding_path = args.output_dir / ("textcnn_dair_v001_padding_summary.csv")

    summary_path = args.output_dir / ("textcnn_dair_v001_interpretation_summary.json")

    filter_frequency.to_csv(
        filter_path,
        index=False,
    )

    span_frequency.to_csv(
        span_path,
        index=False,
    )

    padding_summary.to_csv(
        padding_path,
        index=False,
    )

    cohort_counts = Counter(record["cohort"] for record in records)

    predicted_counts = Counter(record["saved_prediction"] for record in records)

    correct_count = sum(
        record["true_label"] == record["saved_prediction"] for record in records
    )

    padding_values = padding_summary["padding_activation_count"]

    summary = {
        "model_id": records[0]["model_id"],
        "num_examples": len(records),
        "top_spans_per_example": (args.top_spans_per_example),
        "cohort_counts": dict(cohort_counts),
        "predicted_label_counts": dict(predicted_counts),
        "correct_count": int(correct_count),
        "wrong_count": int(len(records) - correct_count),
        "padding": {
            "mean_padding_max_filters": float(padding_values.mean()),
            "median_padding_max_filters": float(padding_values.median()),
            "max_padding_max_filters": int(padding_values.max()),
            "min_padding_max_filters": int(padding_values.min()),
        },
        "top_filter_patterns": (
            top_records(
                filter_frequency,
                n=20,
            )
        ),
        "top_span_patterns": (
            top_records(
                span_frequency,
                n=20,
            )
        ),
    }

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[OK] Interpretation records validated: {len(records)}")

    print(f"[RESULT] Correct examples: {correct_count}")

    print(f"[RESULT] Wrong examples: {len(records) - correct_count}")

    print(f"[RESULT] Mean padding-max filters: {padding_values.mean():.2f}")

    print(f"[RESULT] Median padding-max filters: {padding_values.median():.2f}")

    print(f"[RESULT] Max padding-max filters: {padding_values.max()}")

    print(f"[OK] Filter frequency: {filter_path}")

    print(f"[OK] Span frequency: {span_path}")

    print(f"[OK] Padding summary: {padding_path}")

    print(f"[OK] Interpretation summary: {summary_path}")


if __name__ == "__main__":
    main()
