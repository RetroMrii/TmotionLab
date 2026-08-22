from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_TEXTCNN_PREDICTIONS = Path(
    "ml/artifacts/predictions/textcnn_dair_v001_test_predictions.csv"
)

DEFAULT_COMPARISON = Path("ml/artifacts/comparisons/dair_v001_model_comparison.csv")

DEFAULT_INTERPRETATIONS = Path(
    "ml/artifacts/interpretations/textcnn_dair_v001_interpretations.jsonl"
)

DEFAULT_OUTPUT_DIR = Path("ml/artifacts/errors")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build canonical TmotionLab Error Lab artifacts for TextCNN.")
    )

    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_TEXTCNN_PREDICTIONS,
    )

    parser.add_argument(
        "--comparison",
        type=Path,
        default=DEFAULT_COMPARISON,
    )

    parser.add_argument(
        "--interpretations",
        type=Path,
        default=DEFAULT_INTERPRETATIONS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--high-score-threshold",
        type=float,
        default=0.80,
    )

    return parser.parse_args()


def load_predictions(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"TextCNN predictions not found: {path}")

    frame = pd.read_csv(path)

    required = {
        "example_id",
        "text",
        "true_label",
        "true_label_name",
        "predicted_label",
        "predicted_label_name",
        "top_score",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"TextCNN predictions missing columns: {sorted(missing)}")

    if frame["example_id"].duplicated().any():
        raise ValueError("TextCNN predictions contain duplicate example_id values")

    return frame


def load_comparison(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Comparison artifact not found: {path}")

    frame = pd.read_csv(path)

    required = {
        "example_id",
        "baseline_prediction",
        "baseline_score",
        "baseline_correct",
        "numpy_prediction",
        "numpy_score",
        "numpy_correct",
        "textcnn_prediction",
        "textcnn_score",
        "textcnn_correct",
        "all_models_agree",
        "all_models_correct",
        "all_models_wrong",
        "all_agree_and_wrong",
        "baseline_wrong_textcnn_correct",
        "baseline_correct_textcnn_wrong",
        "numpy_wrong_textcnn_correct",
        "numpy_correct_textcnn_wrong",
        "three_way_disagreement",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"Comparison artifact missing columns: {sorted(missing)}")

    if frame["example_id"].duplicated().any():
        raise ValueError("Comparison artifact contains duplicate example_id values")

    return frame


def load_interpretation_index(
    path: Path,
) -> dict[str, dict]:
    if not path.exists():
        return {}

    index: dict[str, dict] = {}

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
                raise ValueError(
                    f"Invalid interpretation JSON on line {line_number}"
                ) from exc

            example_id = str(record["example_id"])

            if example_id in index:
                raise ValueError(f"Duplicate interpretation example_id: {example_id}")

            index[example_id] = record

    return index


def validate_alignment(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    if len(predictions) != len(comparison):
        raise ValueError(
            "Prediction/comparison row count mismatch: "
            f"{len(predictions)} != {len(comparison)}"
        )

    if not predictions["example_id"].equals(comparison["example_id"]):
        raise ValueError("Prediction/comparison example_id alignment mismatch")

    if not predictions["predicted_label_name"].equals(comparison["textcnn_prediction"]):
        raise ValueError("TextCNN predictions do not match comparison artifact")


def extract_interpretation_fields(
    record: dict | None,
) -> dict:
    if record is None:
        return {
            "interpretation_available": False,
            "interpretation_cohort": None,
            "padding_activation_count": None,
            "top_activation_span_1": None,
            "top_activation_span_2": None,
            "top_activation_span_3": None,
        }

    interpretation = record["interpretation"]

    spans = interpretation.get(
        "top_text_spans",
        [],
    )

    span_texts = [str(span.get("span_text", "")) for span in spans[:3]]

    while len(span_texts) < 3:
        span_texts.append(None)

    return {
        "interpretation_available": True,
        "interpretation_cohort": (record.get("cohort")),
        "padding_activation_count": (interpretation.get("padding_activation_count")),
        "top_activation_span_1": (span_texts[0]),
        "top_activation_span_2": (span_texts[1]),
        "top_activation_span_3": (span_texts[2]),
    }


def build_error_lab_dataset(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    interpretation_index: dict[str, dict],
    *,
    high_score_threshold: float,
) -> pd.DataFrame:
    validate_alignment(
        predictions,
        comparison,
    )

    if not 0.0 <= high_score_threshold <= 1.0:
        raise ValueError("high_score_threshold must be between 0 and 1")

    errors = predictions[
        predictions["predicted_label"] != predictions["true_label"]
    ].copy()

    comparison_columns = [
        "example_id",
        "baseline_prediction",
        "baseline_score",
        "baseline_correct",
        "numpy_prediction",
        "numpy_score",
        "numpy_correct",
        "all_models_agree",
        "all_models_wrong",
        "all_agree_and_wrong",
        "baseline_correct_textcnn_wrong",
        "numpy_correct_textcnn_wrong",
        "three_way_disagreement",
    ]

    errors = errors.merge(
        comparison[comparison_columns],
        on="example_id",
        how="left",
        validate="one_to_one",
    )

    if errors["baseline_prediction"].isna().any():
        raise ValueError("Some TextCNN errors could not be matched to comparison data")

    errors = errors.rename(
        columns={
            "top_score": "textcnn_score",
            "predicted_label": ("textcnn_prediction_id"),
            "predicted_label_name": ("textcnn_prediction"),
        }
    )

    errors["confusion_pair"] = (
        errors["true_label_name"] + " -> " + errors["textcnn_prediction"]
    )

    errors["high_score_error"] = errors["textcnn_score"] >= high_score_threshold

    errors["score_threshold"] = high_score_threshold

    errors["all_three_wrong"] = errors["all_models_wrong"]

    interpretation_rows = []

    for example_id in errors["example_id"].astype(str):
        interpretation_rows.append(
            extract_interpretation_fields(interpretation_index.get(example_id))
        )

    interpretation_frame = pd.DataFrame(
        interpretation_rows,
        index=errors.index,
    )

    errors = pd.concat(
        [
            errors,
            interpretation_frame,
        ],
        axis=1,
    )

    # Analyst categories are intentionally blank.
    # These are hypotheses requiring human review.
    errors["error_category"] = ""
    errors["analyst_notes"] = ""
    errors["review_status"] = "unreviewed"

    errors = errors.sort_values(
        [
            "textcnn_score",
            "example_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    errors.insert(
        0,
        "error_rank",
        range(
            1,
            len(errors) + 1,
        ),
    )

    return errors


def build_confusion_summary(
    errors: pd.DataFrame,
) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame(
            columns=[
                "true_label_name",
                "textcnn_prediction",
                "confusion_pair",
                "count",
                "mean_textcnn_score",
                "max_textcnn_score",
                "high_score_errors",
            ]
        )

    summary = (
        errors.groupby(
            [
                "true_label_name",
                "textcnn_prediction",
                "confusion_pair",
            ],
            as_index=False,
        )
        .agg(
            count=(
                "example_id",
                "count",
            ),
            mean_textcnn_score=(
                "textcnn_score",
                "mean",
            ),
            max_textcnn_score=(
                "textcnn_score",
                "max",
            ),
            high_score_errors=(
                "high_score_error",
                "sum",
            ),
        )
        .sort_values(
            [
                "count",
                "mean_textcnn_score",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    return summary


def build_summary(
    errors: pd.DataFrame,
    confusion_pairs: pd.DataFrame,
    *,
    total_test_examples: int,
    high_score_threshold: float,
) -> dict:
    error_count = len(errors)

    high_score_count = int(errors["high_score_error"].sum())

    interpretation_count = int(errors["interpretation_available"].sum())

    return {
        "model_id": "textcnn_dair_v001",
        "total_test_examples": (total_test_examples),
        "error_count": error_count,
        "correct_count": (total_test_examples - error_count),
        "error_rate": (
            error_count / total_test_examples if total_test_examples else 0.0
        ),
        "high_score_threshold": (high_score_threshold),
        "high_score_error_count": (high_score_count),
        "high_score_error_rate_among_errors": (
            high_score_count / error_count if error_count else 0.0
        ),
        "interpretation_available_count": (interpretation_count),
        "all_models_agree_and_wrong_count": int(errors["all_agree_and_wrong"].sum()),
        "baseline_correct_textcnn_wrong_count": int(
            errors["baseline_correct_textcnn_wrong"].sum()
        ),
        "numpy_correct_textcnn_wrong_count": int(
            errors["numpy_correct_textcnn_wrong"].sum()
        ),
        "three_way_disagreement_count": int(errors["three_way_disagreement"].sum()),
        "num_confusion_pairs": int(len(confusion_pairs)),
        "top_confusion_pairs": (confusion_pairs.head(15).to_dict(orient="records")),
        "error_categories": {
            "status": "manual_review_required",
            "allowed_categories": [
                "semantic_overlap",
                "sarcasm_or_irony",
                "ambiguous_text",
                "rare_emotion_label",
                "weak_context",
                "negation_failure",
                "confidence_overestimation",
                "preprocessing_artifact",
                "other",
            ],
        },
    }


def main() -> None:
    args = parse_args()

    predictions = load_predictions(args.predictions)

    comparison = load_comparison(args.comparison)

    interpretations = load_interpretation_index(args.interpretations)

    errors = build_error_lab_dataset(
        predictions,
        comparison,
        interpretations,
        high_score_threshold=(args.high_score_threshold),
    )

    confusion_pairs = build_confusion_summary(errors)

    summary = build_summary(
        errors,
        confusion_pairs,
        total_test_examples=len(predictions),
        high_score_threshold=(args.high_score_threshold),
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    errors_path = args.output_dir / "textcnn_dair_v001_errors.csv"

    confusion_path = args.output_dir / ("textcnn_dair_v001_confusion_pairs.csv")

    summary_path = args.output_dir / ("textcnn_dair_v001_error_summary.json")

    errors.to_csv(
        errors_path,
        index=False,
        encoding="utf-8",
    )

    confusion_pairs.to_csv(
        confusion_path,
        index=False,
        encoding="utf-8",
    )

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

    print(f"[OK] Test examples validated: {len(predictions)}")

    print(f"[RESULT] TextCNN errors: {len(errors)}")

    print(f"[RESULT] Error rate: {summary['error_rate']:.4f}")

    print(
        "[RESULT] High-score errors "
        f"(>= {args.high_score_threshold:.2f}): "
        f"{summary['high_score_error_count']}"
    )

    print(f"[RESULT] Unique confusion pairs: {len(confusion_pairs)}")

    print(
        "[RESULT] Errors with interpretation data: "
        f"{summary['interpretation_available_count']}"
    )

    print(
        "[RESULT] All models agree and wrong: "
        f"{summary['all_models_agree_and_wrong_count']}"
    )

    print(
        "[RESULT] Baseline correct / TextCNN wrong: "
        f"{summary['baseline_correct_textcnn_wrong_count']}"
    )

    print(f"[OK] Error dataset: {errors_path}")

    print(f"[OK] Confusion pairs: {confusion_path}")

    print(f"[OK] Error summary: {summary_path}")


if __name__ == "__main__":
    main()
