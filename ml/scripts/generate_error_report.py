from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ERRORS = Path("ml/artifacts/errors/textcnn_dair_v001_errors.csv")

DEFAULT_CONFUSIONS = Path("ml/artifacts/errors/textcnn_dair_v001_confusion_pairs.csv")

DEFAULT_SUMMARY = Path("ml/artifacts/errors/textcnn_dair_v001_error_summary.json")

DEFAULT_OUTPUT = Path("reports/error_analysis.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate the TmotionLab benchmark-relative Error Lab report.")
    )

    parser.add_argument(
        "--errors",
        type=Path,
        default=DEFAULT_ERRORS,
    )

    parser.add_argument(
        "--confusions",
        type=Path,
        default=DEFAULT_CONFUSIONS,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV artifact not found: {path}")

    return pd.read_csv(path)


def load_json(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return data


def confusion_table(
    frame: pd.DataFrame,
    *,
    n: int = 15,
) -> str:
    if frame.empty:
        return "_No confusion pairs available._"

    lines = [
        (
            "| True label | Predicted label | Count | "
            "Mean score | Max score | High-score disagreements |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, row in frame.head(n).iterrows():
        lines.append(
            f"| {row['true_label_name']} "
            f"| {row['textcnn_prediction']} "
            f"| {int(row['count'])} "
            f"| {float(row['mean_textcnn_score']):.4f} "
            f"| {float(row['max_textcnn_score']):.4f} "
            f"| {int(row['high_score_errors'])} |"
        )

    return "\n".join(lines)


def example_table(
    frame: pd.DataFrame,
    *,
    n: int = 10,
) -> str:
    if frame.empty:
        return "_No examples available._"

    lines = [
        ("| Example | True | TextCNN | Score | Baseline | NumPy | Text |"),
        "|---|---|---|---:|---|---|---|",
    ]

    for _, row in frame.head(n).iterrows():
        text = str(row["text"]).replace(
            "|",
            "\\|",
        )

        lines.append(
            f"| `{row['example_id']}` "
            f"| {row['true_label_name']} "
            f"| {row['textcnn_prediction']} "
            f"| {float(row['textcnn_score']):.4f} "
            f"| {row['baseline_prediction']} "
            f"| {row['numpy_prediction']} "
            f"| {text} |"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    errors = load_csv(args.errors)

    confusions = load_csv(args.confusions)

    summary = load_json(args.summary)

    if len(errors) != int(summary["error_count"]):
        raise ValueError("Error CSV row count does not match summary")

    high_score = errors[errors["high_score_error"] == True].sort_values(
        "textcnn_score",
        ascending=False,
    )

    shared_wrong = errors[errors["all_agree_and_wrong"] == True].sort_values(
        "textcnn_score",
        ascending=False,
    )

    baseline_regressions = errors[
        errors["baseline_correct_textcnn_wrong"] == True
    ].sort_values(
        "textcnn_score",
        ascending=False,
    )

    three_way = errors[errors["three_way_disagreement"] == True].sort_values(
        "textcnn_score",
        ascending=False,
    )

    interpretation_examples = errors[
        errors["interpretation_available"] == True
    ].sort_values(
        "textcnn_score",
        ascending=False,
    )

    report_parts = [
        "# TmotionLab Error Lab Analysis",
        "",
        "## Purpose",
        "",
        (
            "This report analyzes TextCNN predictions that "
            "disagree with the DAIR.AI Emotion test labels."
        ),
        "",
        (
            "The analysis is benchmark-relative: a row is "
            "treated as an evaluation error when the model "
            "prediction differs from the dataset label."
        ),
        "",
        (
            "**Important caveat:** this does not imply that "
            "every disagreement is a clear semantic model "
            "failure. Some examples may reflect ambiguous "
            "emotion boundaries, mixed emotions, limited "
            "context, or annotation noise. No systematic "
            "manual adjudication or dataset relabeling was "
            "performed in this version."
        ),
        "",
        "## Overall Error Summary",
        "",
        (f"- **Model:** `{summary['model_id']}`"),
        (f"- **Test examples:** {summary['total_test_examples']}"),
        (f"- **Benchmark disagreements:** {summary['error_count']}"),
        (f"- **Benchmark error rate:** {summary['error_rate']:.4f}"),
        (f"- **High-score threshold:** {summary['high_score_threshold']:.2f}"),
        (f"- **High-score disagreements:** {summary['high_score_error_count']}"),
        (f"- **Unique confusion pairs:** {summary['num_confusion_pairs']}"),
        (
            f"- **Interpretation available:** "
            f"{summary['interpretation_available_count']}"
        ),
        "",
        "## Dominant Confusion Pairs",
        "",
        confusion_table(
            confusions,
            n=15,
        ),
        "",
        (
            "These counts describe disagreement with the "
            "benchmark label. They do not establish that the "
            "model prediction is semantically unreasonable."
        ),
        "",
        "## High-Score Benchmark Disagreements",
        "",
        (
            "These are disagreements where the TextCNN "
            f"softmax score is at least "
            f"{summary['high_score_threshold']:.2f}."
        ),
        "",
        ("The score is not treated as calibrated confidence."),
        "",
        example_table(
            high_score,
            n=12,
        ),
        "",
        "## Shared Disagreements Across All Models",
        "",
        (
            f"All three model families produce the same "
            f"wrong benchmark prediction on "
            f"**{summary['all_models_agree_and_wrong_count']}** "
            f"examples."
        ),
        "",
        (
            "These examples are especially useful because "
            "Logistic Regression, the NumPy MLP, and TextCNN "
            "arrive at the same alternative label despite "
            "using different representations and architectures."
        ),
        "",
        (
            "Cross-model agreement is evidence of a stable "
            "pattern, not proof that the benchmark annotation "
            "is wrong."
        ),
        "",
        example_table(
            shared_wrong,
            n=12,
        ),
        "",
        "## TextCNN Regressions Relative to Baseline",
        "",
        (
            f"The Logistic Regression baseline is benchmark-"
            f"correct while TextCNN is benchmark-wrong on "
            f"**{summary['baseline_correct_textcnn_wrong_count']}** "
            f"examples."
        ),
        "",
        (
            "These cases are useful because they show where "
            "the more complex sequence model performs worse "
            "than the simpler lexical baseline."
        ),
        "",
        example_table(
            baseline_regressions,
            n=12,
        ),
        "",
        "## Three-Way Disagreements",
        "",
        (
            f"TextCNN, NumPy MLP, and Logistic Regression "
            f"all choose different labels on "
            f"**{summary['three_way_disagreement_count']}** "
            f"TextCNN benchmark-error examples."
        ),
        "",
        example_table(
            three_way,
            n=10,
        ),
        "",
        "## Interpretation Coverage",
        "",
        (
            f"Interpretation metadata is available for "
            f"**{summary['interpretation_available_count']}** "
            f"of the {summary['error_count']} benchmark "
            f"disagreements."
        ),
        "",
        (
            "This coverage comes from the P7 held-out "
            "interpretation sample and is intentionally "
            "partial."
        ),
        "",
        example_table(
            interpretation_examples,
            n=10,
        ),
        "",
        "## Error Lab Interpretation Rules",
        "",
        ("The current Error Lab uses the following terminology:"),
        "",
        (
            "- **Benchmark disagreement:** model prediction "
            "differs from the dataset label."
        ),
        (
            "- **Evaluation error:** equivalent benchmark-"
            "relative term used for metric analysis."
        ),
        (
            "- **High-score disagreement:** benchmark "
            "disagreement with a high raw softmax score."
        ),
        "",
        ("The following claims are deliberately avoided without human adjudication:"),
        "",
        "- the model is definitely semantically wrong;",
        "- the dataset annotation is definitely wrong;",
        "- a high score proves model confidence is calibrated;",
        "- all shared-model disagreements are mislabeled data;",
        "",
        "## Deferred Analysis",
        "",
        ("The following extensions are intentionally deferred:"),
        "",
        "- systematic manual review of all disagreements;",
        "- analyst-authored error taxonomy;",
        "- dataset relabeling or annotation-quality study;",
        "- multi-reviewer agreement analysis;",
        "- calibration analysis;",
        "",
        "## Main Takeaway",
        "",
        (
            "The TextCNN achieves strong benchmark performance, "
            "but its remaining disagreements are not uniform."
        ),
        "",
        (
            "Some are model-specific regressions, some are "
            "shared across all three architectures, and some "
            "occur with very high output scores."
        ),
        "",
        (
            "TmotionLab therefore treats error analysis as "
            "inspection of model behavior relative to the "
            "benchmark rather than assuming that every "
            "label disagreement has a single obvious cause."
        ),
    ]

    report = "\n".join(report_parts)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        report,
        encoding="utf-8",
    )

    print(f"[OK] Error Lab report: {args.output}")

    print(f"[RESULT] Benchmark disagreements: {summary['error_count']}")

    print(f"[RESULT] High-score disagreements: {summary['high_score_error_count']}")

    print(
        "[RESULT] All-model shared disagreements: "
        f"{summary['all_models_agree_and_wrong_count']}"
    )

    print(
        "[RESULT] Baseline correct / TextCNN wrong: "
        f"{summary['baseline_correct_textcnn_wrong_count']}"
    )


if __name__ == "__main__":
    main()
