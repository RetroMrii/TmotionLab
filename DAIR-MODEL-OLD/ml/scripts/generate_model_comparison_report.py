from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the TmotionLab model-comparison "
            "Markdown report from saved artifacts."
        )
    )

    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("ml/artifacts/metrics/baseline_tfidf_dair_v001_metrics.json"),
    )

    parser.add_argument(
        "--numpy-metrics",
        type=Path,
        default=Path("ml/artifacts/metrics/numpy_mlp_dair_v002_metrics.json"),
    )

    parser.add_argument(
        "--textcnn-metrics",
        type=Path,
        default=Path("ml/artifacts/metrics/textcnn_dair_v001_metrics.json"),
    )

    parser.add_argument(
        "--comparison-summary",
        type=Path,
        default=Path("ml/artifacts/comparisons/dair_v001_comparison_summary.json"),
    )

    parser.add_argument(
        "--behavior-summary",
        type=Path,
        default=Path("ml/artifacts/comparisons/dair_v001_behavior_summary.json"),
    )

    parser.add_argument(
        "--shared-errors",
        type=Path,
        default=Path("ml/artifacts/comparisons/dair_v001_shared_errors.csv"),
    )

    parser.add_argument(
        "--disagreements",
        type=Path,
        default=Path("ml/artifacts/comparisons/dair_v001_three_way_disagreements.csv"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/model_comparison.md"),
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in: {path}")

    return data


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    return pd.read_csv(path)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def metric_row(
    name: str,
    metrics: dict,
) -> str:
    test = metrics["test"]

    return (
        f"| {name} "
        f"| {test['accuracy']:.4f} "
        f"| {test['macro_f1']:.4f} "
        f"| {test['weighted_f1']:.4f} |"
    )


def markdown_class_table(
    records: list[dict],
) -> str:
    if not records:
        return "_No examples in this cohort._"

    lines = [
        "| True class | Count |",
        "|---|---:|",
    ]

    for record in records:
        lines.append(f"| {record['true_label_name']} | {record['count']} |")

    return "\n".join(lines)


def markdown_confusion_table(
    records: list[dict],
    *,
    prediction_key: str,
) -> str:
    if not records:
        return "_No confusion pairs in this cohort._"

    lines = [
        "| True class | Predicted class | Count |",
        "|---|---|---:|",
    ]

    for record in records[:15]:
        lines.append(
            f"| {record['true_label_name']} "
            f"| {record[prediction_key]} "
            f"| {record['count']} |"
        )

    return "\n".join(lines)


def example_table(
    frame: pd.DataFrame,
    *,
    max_rows: int = 10,
) -> str:
    if frame.empty:
        return "_No examples available._"

    required = [
        "example_id",
        "true_label_name",
        "baseline_prediction",
        "numpy_prediction",
        "textcnn_prediction",
        "text",
    ]

    missing = set(required) - set(frame.columns)

    if missing:
        raise ValueError(f"Example artifact missing columns: {sorted(missing)}")

    lines = [
        ("| Example | True | Baseline | NumPy MLP | TextCNN | Text |"),
        "|---|---|---|---|---|---|",
    ]

    for _, row in frame.head(max_rows).iterrows():
        text = str(row["text"]).replace(
            "|",
            "\\|",
        )

        lines.append(
            f"| `{row['example_id']}` "
            f"| {row['true_label_name']} "
            f"| {row['baseline_prediction']} "
            f"| {row['numpy_prediction']} "
            f"| {row['textcnn_prediction']} "
            f"| {text} |"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    baseline = load_json(args.baseline_metrics)

    numpy_model = load_json(args.numpy_metrics)

    textcnn = load_json(args.textcnn_metrics)

    comparison = load_json(args.comparison_summary)

    behavior = load_json(args.behavior_summary)

    shared_errors = load_csv(args.shared_errors)

    disagreements = load_csv(args.disagreements)

    baseline_test = baseline["test"]
    numpy_test = numpy_model["test"]
    textcnn_test = textcnn["test"]

    accuracy_gain_vs_baseline = textcnn_test["accuracy"] - baseline_test["accuracy"]

    macro_gain_vs_baseline = textcnn_test["macro_f1"] - baseline_test["macro_f1"]

    weighted_gain_vs_baseline = (
        textcnn_test["weighted_f1"] - baseline_test["weighted_f1"]
    )

    macro_gain_vs_numpy = textcnn_test["macro_f1"] - numpy_test["macro_f1"]

    agreement = comparison["agreement"]
    counts = comparison["counts"]

    fixes = behavior["cohorts"]["textcnn_fixes_baseline"]

    regressions = behavior["cohorts"]["baseline_beats_textcnn"]

    shared = behavior["cohorts"]["all_agree_and_wrong"]

    three_way = behavior["cohorts"]["three_way_disagreement"]

    report = f"""# TmotionLab Model Comparison

## Purpose

This report compares the three promoted TmotionLab emotion-classification models on the same 2,000-example DAIR.AI Emotion test split.

The comparison is based on identical `example_id` values, text, and ground-truth labels across all model prediction artifacts.

The three models are:

1. TF-IDF + Logistic Regression baseline.
2. From-scratch NumPy MLP using TF-IDF input features.
3. PyTorch TextCNN using learned embeddings and convolutional text features.

## Test Metrics

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
{metric_row("TF-IDF + Logistic Regression", baseline)}
{metric_row("NumPy MLP v002", numpy_model)}
{metric_row("TextCNN v001", textcnn)}

TextCNN improves over the Logistic Regression baseline by:

- **Accuracy:** {accuracy_gain_vs_baseline:+.4f}
- **Macro F1:** {macro_gain_vs_baseline:+.4f}
- **Weighted F1:** {weighted_gain_vs_baseline:+.4f}

TextCNN also improves macro F1 over the NumPy MLP by **{macro_gain_vs_numpy:+.4f}**.

## Agreement Analysis

Across the {comparison["num_examples"]} aligned test examples:

- all three models agree on **{pct(agreement["all_three_rate"])}** of examples;
- Logistic Regression and NumPy MLP agree on **{
        pct(agreement["baseline_numpy_rate"])
    }**;
- Logistic Regression and TextCNN agree on **{
        pct(agreement["baseline_textcnn_rate"])
    }**;
- NumPy MLP and TextCNN agree on **{pct(agreement["numpy_textcnn_rate"])}**;
- all three models produce different predictions on **{
        counts["three_way_disagreement"]
    }** examples.

High agreement does not imply correctness. There are **{
        counts["all_agree_and_wrong"]
    }** examples where all three models make the same incorrect prediction.

## TextCNN Gains Over the Baseline

TextCNN corrects **{
        counts["baseline_wrong_textcnn_correct"]
    }** examples that the Logistic Regression baseline predicts incorrectly.

The baseline remains correct while TextCNN fails on **{
        counts["baseline_correct_textcnn_wrong"]
    }** examples.

This produces a net gain of:

**{
        counts["baseline_wrong_textcnn_correct"]
        - counts["baseline_correct_textcnn_wrong"]
    } test examples.**

### TextCNN Fixes by True Class

{markdown_class_table(fixes["by_true_class"])}

### Baseline Confusions Repaired by TextCNN

{
        markdown_confusion_table(
            fixes["baseline_confusions_fixed"],
            prediction_key="baseline_prediction",
        )
    }

These cohorts are more informative than aggregate accuracy alone because they identify where the sequence-aware TextCNN changes behavior relative to the lexical TF-IDF baseline.

## TextCNN Regressions

TextCNN introduces **{
        counts["baseline_correct_textcnn_wrong"]
    }** errors on examples that the baseline classified correctly.

### Regressions by True Class

{markdown_class_table(regressions["by_true_class"])}

### TextCNN Regression Confusions

{
        markdown_confusion_table(
            regressions["textcnn_regressions"],
            prediction_key="textcnn_prediction",
        )
    }

These errors prevent the comparison from being reduced to the statement that TextCNN is universally better. The model is better in aggregate but still loses information or makes poorer decisions on some examples.

## Errors Shared by All Models

All three models agree on the same wrong prediction for **{
        shared["count"]
    }** examples, representing **{pct(shared["rate"])}** of the test split.

### Shared Errors by True Class

{markdown_class_table(shared["by_true_class"])}

### Most Common Shared Confusions

{
        markdown_confusion_table(
            shared["shared_confusions"],
            prediction_key="textcnn_prediction",
        )
    }

### Example Shared Errors

{example_table(shared_errors)}

These cases are particularly important for the later Error Lab. Agreement across three structurally different models suggests that some examples may involve difficult class boundaries, semantic overlap, weak context, or dataset-label ambiguity.

Those possibilities are hypotheses only. A prediction disagreement with the dataset label is not sufficient evidence that the dataset label itself is incorrect.

## Three-Way Disagreement

Only **{three_way["count"]}** examples produce three different model predictions.

### Three-Way Disagreements by True Class

{markdown_class_table(three_way["by_true_class"])}

### Example Three-Way Disagreements

{example_table(disagreements)}

These examples provide a compact high-value cohort for manual inspection because each model family interprets the same text differently.

## Main Findings

1. **Model complexity produces measurable gains.**  
   Performance improves from Logistic Regression to the NumPy MLP and again from the NumPy MLP to TextCNN.

2. **TextCNN's improvement is behaviorally observable, not only metric-level.**  
   It repairs {
        counts["baseline_wrong_textcnn_correct"]
    } baseline errors while introducing {
        counts["baseline_correct_textcnn_wrong"]
    } regressions.

3. **Most examples are not useful for distinguishing model behavior.**  
   The all-model agreement rate is {
        pct(agreement["all_three_rate"])
    }, so the most informative analysis lies in disagreement and shared-error cohorts.

4. **Shared errors remain important despite the stronger neural model.**  
   All three systems agree and fail on {counts["all_agree_and_wrong"]} examples.

5. **The comparison supports an Error Lab rather than a leaderboard-only presentation.**  
   The meaningful question is not only which model has the highest score, but which language patterns and label boundaries cause models to behave differently.

## Limitations

- The models use different input representations, which is intentional but means internal feature spaces are not directly comparable.
- Raw probability-like scores should not be compared across model families as though they were equally calibrated.
- The analysis is based on one verified held-out dataset split.
- Dataset labels are treated as evaluation targets, not universal psychological truth.
- Some apparent errors may reflect ambiguity or semantic overlap rather than a simple model failure.
- Interpretation of TextCNN internal activations is handled separately and should not be confused with causal explanations of emotion.

## Next Step

The next phase is **P7 — TextCNN Interpretation Lab**.

That phase will inspect convolutional activation regions and map high-activation filters back to exact input token spans, allowing TmotionLab to investigate why particular local text patterns strongly activate the production model.
"""

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        report,
        encoding="utf-8",
    )

    print(f"[OK] Model comparison report: {args.output}")

    print(f"[RESULT] TextCNN macro F1 gain vs baseline: {macro_gain_vs_baseline:+.4f}")

    print(
        "[RESULT] TextCNN net corrected "
        "examples vs baseline: "
        f"{counts['baseline_wrong_textcnn_correct'] - counts['baseline_correct_textcnn_wrong']}"
    )


if __name__ == "__main__":
    main()
