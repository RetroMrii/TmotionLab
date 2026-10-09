from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODEL_FILES = {
    "baseline": (
        "ml/artifacts/predictions/baseline_tfidf_dair_v001_test_predictions.csv"
    ),
    "numpy": ("ml/artifacts/predictions/numpy_mlp_dair_v002_test_predictions.csv"),
    "textcnn": ("ml/artifacts/predictions/textcnn_dair_v001_test_predictions.csv"),
}


REQUIRED_COLUMNS = {
    "example_id",
    "text",
    "true_label",
    "true_label_name",
    "predicted_label",
    "predicted_label_name",
    "top_score",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare promoted TmotionLab models on the same held-out examples."
        )
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(MODEL_FILES["baseline"]),
    )

    parser.add_argument(
        "--numpy",
        type=Path,
        default=Path(MODEL_FILES["numpy"]),
    )

    parser.add_argument(
        "--textcnn",
        type=Path,
        default=Path(MODEL_FILES["textcnn"]),
    )

    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("ml/artifacts/comparisons/dair_v001_model_comparison.csv"),
    )

    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path("ml/artifacts/comparisons/dair_v001_comparison_summary.json"),
    )

    return parser.parse_args()


def load_predictions(
    path: Path,
    *,
    model_name: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{model_name} predictions not found: {path}")

    frame = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(frame.columns)

    if missing:
        raise ValueError(f"{model_name}: missing columns: {sorted(missing)}")

    if frame["example_id"].duplicated().any():
        raise ValueError(f"{model_name}: duplicate example_id values")

    return frame


def validate_alignment(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    candidate_name: str,
) -> None:
    if len(reference) != len(candidate):
        raise ValueError(
            f"{candidate_name}: row count mismatch. "
            f"Expected {len(reference)}, "
            f"found {len(candidate)}."
        )

    if not reference["example_id"].equals(candidate["example_id"]):
        raise ValueError(
            f"{candidate_name}: example_id order does not match the reference model."
        )

    if not reference["true_label"].equals(candidate["true_label"]):
        raise ValueError(
            f"{candidate_name}: true_label values do not match the reference model."
        )

    if not reference["true_label_name"].equals(candidate["true_label_name"]):
        raise ValueError(
            f"{candidate_name}: true_label_name values "
            "do not match the reference model."
        )

    if not reference["text"].equals(candidate["text"]):
        raise ValueError(
            f"{candidate_name}: input text does not match the reference model."
        )


def add_model_columns(
    output: pd.DataFrame,
    source: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    output[f"{prefix}_prediction"] = source["predicted_label_name"]

    output[f"{prefix}_prediction_id"] = source["predicted_label"]

    output[f"{prefix}_score"] = source["top_score"]

    output[f"{prefix}_correct"] = source["predicted_label"] == source["true_label"]


def build_comparison(
    baseline: pd.DataFrame,
    numpy_model: pd.DataFrame,
    textcnn: pd.DataFrame,
) -> pd.DataFrame:
    output = baseline[
        [
            "example_id",
            "text",
            "true_label",
            "true_label_name",
        ]
    ].copy()

    add_model_columns(
        output,
        baseline,
        prefix="baseline",
    )

    add_model_columns(
        output,
        numpy_model,
        prefix="numpy",
    )

    add_model_columns(
        output,
        textcnn,
        prefix="textcnn",
    )

    output["all_models_agree"] = (
        output["baseline_prediction"] == output["numpy_prediction"]
    ) & (output["numpy_prediction"] == output["textcnn_prediction"])

    output["all_models_correct"] = (
        output["baseline_correct"] & output["numpy_correct"] & output["textcnn_correct"]
    )

    output["all_models_wrong"] = (
        ~output["baseline_correct"]
        & ~output["numpy_correct"]
        & ~output["textcnn_correct"]
    )

    output["all_agree_and_correct"] = (
        output["all_models_agree"] & output["all_models_correct"]
    )

    output["all_agree_and_wrong"] = (
        output["all_models_agree"] & output["all_models_wrong"]
    )

    output["baseline_wrong_textcnn_correct"] = (
        ~output["baseline_correct"] & output["textcnn_correct"]
    )

    output["baseline_correct_textcnn_wrong"] = (
        output["baseline_correct"] & ~output["textcnn_correct"]
    )

    output["numpy_wrong_textcnn_correct"] = (
        ~output["numpy_correct"] & output["textcnn_correct"]
    )

    output["numpy_correct_textcnn_wrong"] = (
        output["numpy_correct"] & ~output["textcnn_correct"]
    )

    output["baseline_wrong_numpy_correct"] = (
        ~output["baseline_correct"] & output["numpy_correct"]
    )

    output["baseline_correct_numpy_wrong"] = (
        output["baseline_correct"] & ~output["numpy_correct"]
    )

    output["three_way_disagreement"] = (
        output[
            [
                "baseline_prediction",
                "numpy_prediction",
                "textcnn_prediction",
            ]
        ].nunique(axis=1)
        == 3
    )

    return output


def pairwise_agreement(
    frame: pd.DataFrame,
    first: str,
    second: str,
) -> float:
    return float((frame[f"{first}_prediction"] == frame[f"{second}_prediction"]).mean())


def build_summary(
    comparison: pd.DataFrame,
) -> dict:
    total = len(comparison)

    return {
        "num_examples": total,
        "models": {
            "baseline": "baseline_tfidf_dair_v001",
            "numpy": "numpy_mlp_dair_v002",
            "textcnn": "textcnn_dair_v001",
        },
        "accuracy_from_saved_predictions": {
            "baseline": float(comparison["baseline_correct"].mean()),
            "numpy": float(comparison["numpy_correct"].mean()),
            "textcnn": float(comparison["textcnn_correct"].mean()),
        },
        "agreement": {
            "all_three_rate": float(comparison["all_models_agree"].mean()),
            "baseline_numpy_rate": (
                pairwise_agreement(
                    comparison,
                    "baseline",
                    "numpy",
                )
            ),
            "baseline_textcnn_rate": (
                pairwise_agreement(
                    comparison,
                    "baseline",
                    "textcnn",
                )
            ),
            "numpy_textcnn_rate": (
                pairwise_agreement(
                    comparison,
                    "numpy",
                    "textcnn",
                )
            ),
            "three_way_disagreement_rate": float(
                comparison["three_way_disagreement"].mean()
            ),
        },
        "counts": {
            "all_models_agree": int(comparison["all_models_agree"].sum()),
            "all_models_correct": int(comparison["all_models_correct"].sum()),
            "all_models_wrong": int(comparison["all_models_wrong"].sum()),
            "all_agree_and_correct": int(comparison["all_agree_and_correct"].sum()),
            "all_agree_and_wrong": int(comparison["all_agree_and_wrong"].sum()),
            "baseline_wrong_textcnn_correct": int(
                comparison["baseline_wrong_textcnn_correct"].sum()
            ),
            "baseline_correct_textcnn_wrong": int(
                comparison["baseline_correct_textcnn_wrong"].sum()
            ),
            "numpy_wrong_textcnn_correct": int(
                comparison["numpy_wrong_textcnn_correct"].sum()
            ),
            "numpy_correct_textcnn_wrong": int(
                comparison["numpy_correct_textcnn_wrong"].sum()
            ),
            "baseline_wrong_numpy_correct": int(
                comparison["baseline_wrong_numpy_correct"].sum()
            ),
            "baseline_correct_numpy_wrong": int(
                comparison["baseline_correct_numpy_wrong"].sum()
            ),
            "three_way_disagreement": int(comparison["three_way_disagreement"].sum()),
        },
    }


def main() -> None:
    args = parse_args()

    baseline = load_predictions(
        args.baseline,
        model_name="baseline",
    )

    numpy_model = load_predictions(
        args.numpy,
        model_name="numpy",
    )

    textcnn = load_predictions(
        args.textcnn,
        model_name="textcnn",
    )

    validate_alignment(
        baseline,
        numpy_model,
        candidate_name="numpy",
    )

    validate_alignment(
        baseline,
        textcnn,
        candidate_name="textcnn",
    )

    print(f"[OK] Model prediction files aligned: {len(baseline)} examples")

    comparison = build_comparison(
        baseline,
        numpy_model,
        textcnn,
    )

    summary = build_summary(comparison)

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output_summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        args.output_csv,
        index=False,
        encoding="utf-8",
    )

    with args.output_summary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(
        "[RESULT] All-model agreement rate: "
        f"{summary['agreement']['all_three_rate']:.4f}"
    )

    print(
        "[RESULT] Baseline -> TextCNN fixes: "
        f"{summary['counts']['baseline_wrong_textcnn_correct']}"
    )

    print(
        "[RESULT] Baseline correct / TextCNN wrong: "
        f"{summary['counts']['baseline_correct_textcnn_wrong']}"
    )

    print(
        "[RESULT] NumPy -> TextCNN fixes: "
        f"{summary['counts']['numpy_wrong_textcnn_correct']}"
    )

    print(f"[RESULT] All agree and wrong: {summary['counts']['all_agree_and_wrong']}")

    print(
        "[RESULT] Three-way disagreements: "
        f"{summary['counts']['three_way_disagreement']}"
    )

    print(f"[OK] Comparison CSV: {args.output_csv}")

    print(f"[OK] Comparison summary: {args.output_summary}")


if __name__ == "__main__":
    main()
