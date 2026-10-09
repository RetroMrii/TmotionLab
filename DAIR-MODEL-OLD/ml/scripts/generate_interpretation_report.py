from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Generate the TmotionLab TextCNN interpretation report.")
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "ml/artifacts/interpretations/textcnn_dair_v001_interpretation_summary.json"
        ),
    )

    parser.add_argument(
        "--filter-frequency",
        type=Path,
        default=Path(
            "ml/artifacts/interpretations/textcnn_dair_v001_filter_frequency.csv"
        ),
    )

    parser.add_argument(
        "--span-frequency",
        type=Path,
        default=Path(
            "ml/artifacts/interpretations/textcnn_dair_v001_span_frequency.csv"
        ),
    )

    parser.add_argument(
        "--padding-summary",
        type=Path,
        default=Path(
            "ml/artifacts/interpretations/textcnn_dair_v001_padding_summary.csv"
        ),
    )

    parser.add_argument(
        "--interpretations",
        type=Path,
        default=Path(
            "ml/artifacts/interpretations/textcnn_dair_v001_interpretations.jsonl"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/interpretation_notes.md"),
    )

    return parser.parse_args()


def load_json(path: Path) -> dict:
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


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV artifact not found: {path}")

    return pd.read_csv(path)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL artifact not found: {path}")

    records = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            stripped = line.strip()

            if stripped:
                records.append(json.loads(stripped))

    if not records:
        raise ValueError("Interpretation JSONL contains no records")

    return records


def markdown_filter_table(
    frame: pd.DataFrame,
    *,
    n: int = 15,
) -> str:
    if frame.empty:
        return "_No filter-frequency records available._"

    lines = [
        (
            "| Kernel | Filter | Predicted class | "
            "Unique examples | Mean activation | Mean rank |"
        ),
        "|---:|---:|---|---:|---:|---:|",
    ]

    for _, row in frame.head(n).iterrows():
        lines.append(
            f"| {int(row['kernel_size'])} "
            f"| {int(row['filter_index'])} "
            f"| {row['predicted_label']} "
            f"| {int(row['unique_examples'])} "
            f"| {float(row['mean_activation']):.4f} "
            f"| {float(row['mean_rank']):.2f} |"
        )

    return "\n".join(lines)


def markdown_span_table(
    frame: pd.DataFrame,
    *,
    n: int = 20,
) -> str:
    if frame.empty:
        return "_No span-frequency records available._"

    lines = [
        ("| Span | Predicted class | Unique examples | Mean activation |"),
        "|---|---|---:|---:|",
    ]

    for _, row in frame.head(n).iterrows():
        span = str(row["span_text_normalized"]).replace(
            "|",
            "\\|",
        )

        lines.append(
            f"| `{span}` "
            f"| {row['predicted_label']} "
            f"| {int(row['unique_examples'])} "
            f"| {float(row['mean_activation']):.4f} |"
        )

    return "\n".join(lines)


def cohort_padding_table(
    frame: pd.DataFrame,
) -> str:
    grouped = (
        frame.groupby("cohort")["padding_activation_count"]
        .agg(
            [
                "mean",
                "median",
                "max",
            ]
        )
        .reset_index()
    )

    lines = [
        ("| Cohort | Mean padding maxima | Median | Max |"),
        "|---|---:|---:|---:|",
    ]

    for _, row in grouped.iterrows():
        lines.append(
            f"| {row['cohort']} "
            f"| {float(row['mean']):.2f} "
            f"| {float(row['median']):.2f} "
            f"| {int(row['max'])} |"
        )

    return "\n".join(lines)


def select_example(
    records: list[dict],
    *,
    cohort: str,
) -> dict | None:
    matches = [record for record in records if record["cohort"] == cohort]

    if not matches:
        return None

    if cohort.endswith("high_score"):
        return max(
            matches,
            key=lambda item: item["saved_top_score"],
        )

    return min(
        matches,
        key=lambda item: item["saved_top_score"],
    )


def example_section(
    record: dict | None,
    *,
    title: str,
) -> str:
    if record is None:
        return f"### {title}\n\n_No example available._"

    interpretation = record["interpretation"]

    lines = [
        f"### {title}",
        "",
        f"- **Example ID:** `{record['example_id']}`",
        f"- **True label:** {record['true_label']}",
        f"- **Prediction:** {record['saved_prediction']}",
        (f"- **Prediction score:** {record['saved_top_score']:.4f}"),
        (
            "- **Padding-overlap filter maxima:** "
            f"{interpretation['padding_activation_count']}"
        ),
        "",
        "**Text**",
        "",
        f"> {record['text']}",
        "",
        "**Top valid activation spans**",
        "",
        "| Rank | Kernel | Filter | Activation | Span |",
        "|---:|---:|---:|---:|---|",
    ]

    for rank, span in enumerate(
        interpretation["top_text_spans"][:5],
        start=1,
    ):
        span_text = str(span["span_text"]).replace(
            "|",
            "\\|",
        )

        lines.append(
            f"| {rank} "
            f"| {span['kernel_size']} "
            f"| {span['filter_index']} "
            f"| {span['activation']:.4f} "
            f"| `{span_text}` |"
        )

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    summary = load_json(args.summary)

    filter_frequency = load_csv(args.filter_frequency)

    span_frequency = load_csv(args.span_frequency)

    padding_summary = load_csv(args.padding_summary)

    records = load_jsonl(args.interpretations)

    num_examples = int(summary["num_examples"])

    if num_examples != len(records):
        raise ValueError("Summary example count does not match interpretation JSONL")

    correct_high = select_example(
        records,
        cohort="correct_high_score",
    )

    correct_low = select_example(
        records,
        cohort="correct_low_score",
    )

    wrong_high = select_example(
        records,
        cohort="wrong_high_score",
    )

    wrong_low = select_example(
        records,
        cohort="wrong_low_score",
    )

    padding = summary["padding"]

    report_parts = [
        "# TmotionLab TextCNN Interpretation Notes",
        "",
        "## Purpose",
        "",
        (
            "This report documents the first validated interpretation "
            "layer for `textcnn_dair_v001`."
        ),
        "",
        (
            "The interpretation system exposes high-activation "
            "convolutional regions from the trained TextCNN and maps "
            "their activation positions back to exact token spans "
            "in the input text."
        ),
        "",
        (
            "These outputs are **model-internal evidence**. They are "
            "not causal explanations of a person's emotion and should "
            "not be interpreted as psychological ground truth."
        ),
        "",
        "## Interpretation Method",
        "",
        ("The TextCNN architecture processes token sequences through:"),
        "",
        "```text",
        "token IDs",
        "-> embedding layer",
        "-> Conv1D filters with kernel sizes 3, 4, and 5",
        "-> ReLU",
        "-> global max pooling",
        "-> concatenation",
        "-> classifier",
        "```",
        "",
        (
            "For each convolutional filter, the interpretation pipeline "
            "records the position with the largest post-ReLU activation."
        ),
        "",
        "The corresponding token window is reconstructed using:",
        "",
        "- kernel size;",
        "- filter index;",
        "- activation value;",
        "- start token index;",
        "- end token index;",
        "- exact token span.",
        "",
        ("The model itself is not retrained or modified for interpretation."),
        "",
        "## Validation Sample",
        "",
        (
            f"The interpretation validation artifact contains "
            f"**{num_examples}** held-out examples."
        ),
        "",
        "Cohorts:",
        "",
        "- correct, high-score predictions;",
        "- correct, low-score predictions;",
        "- wrong, high-score predictions;",
        "- wrong, low-score predictions.",
        "",
        "Observed totals:",
        "",
        f"- **Correct examples:** {summary['correct_count']}",
        f"- **Wrong examples:** {summary['wrong_count']}",
        "",
        (
            "The sample is deliberately balanced between correct and "
            "incorrect predictions so that interpretation behavior is "
            "inspected on failures as well as successes."
        ),
        "",
        "## Padding-Overlap Behavior",
        "",
        (
            "A convolutional filter can reach its raw maximum on a "
            "window that overlaps right-padding."
        ),
        "",
        (
            "This matters because the TextCNN uses global max pooling "
            "and convolutional bias terms. A high activation is therefore "
            "not automatically guaranteed to correspond entirely to "
            "real text."
        ),
        "",
        "Across the interpretation sample:",
        "",
        (
            "- **Mean filters with padding-overlap maxima:** "
            f"{padding['mean_padding_max_filters']:.2f}"
        ),
        (f"- **Median:** {padding['median_padding_max_filters']:.2f}"),
        (f"- **Minimum:** {padding['min_padding_max_filters']}"),
        (f"- **Maximum:** {padding['max_padding_max_filters']}"),
        "",
        (
            "User-visible interpretation spans exclude windows that "
            "overlap padding, but the raw padding-overlap count is "
            "preserved as diagnostic metadata."
        ),
        "",
        "### Padding Behavior by Cohort",
        "",
        cohort_padding_table(padding_summary),
        "",
        (
            "A higher padding-overlap count in one cohort does not "
            "establish that padding causes prediction errors. It only "
            "identifies a behavior worth further investigation."
        ),
        "",
        "## Frequently Observed Filters",
        "",
        (
            "The following filters appear most frequently among the "
            "top valid activation spans in the sampled examples:"
        ),
        "",
        markdown_filter_table(filter_frequency),
        "",
        (
            "These observations do **not** justify naming a filter as "
            "a permanent emotion detector."
        ),
        "",
        (
            "For example, if a filter frequently appears among examples "
            "predicted as `fear`, the supported claim is:"
        ),
        "",
        (
            "> The filter frequently appears among the strongest "
            "activation spans for sampled examples predicted as fear."
        ),
        "",
        "The unsupported claim would be:",
        "",
        "> This is a fear filter.",
        "",
        (
            "Class specificity would require broader controlled analysis "
            "across the full held-out set."
        ),
        "",
        "## Frequently Observed Spans",
        "",
        (
            "The most repeated normalized text spans among the sampled "
            "top activations are:"
        ),
        "",
        markdown_span_table(span_frequency),
        "",
        (
            "Repeated generic patterns such as `i feel`, `i am`, or "
            "similar dataset-specific constructions should be treated "
            "cautiously. Their presence may indicate that filters learn "
            "recurring dataset syntax rather than purely emotion-specific "
            "semantic cues."
        ),
        "",
        "## Representative Examples",
        "",
        example_section(
            correct_high,
            title="Correct High-Score Prediction",
        ),
        "",
        example_section(
            correct_low,
            title="Correct Lower-Score Prediction",
        ),
        "",
        example_section(
            wrong_high,
            title="Wrong High-Score Prediction",
        ),
        "",
        example_section(
            wrong_low,
            title="Wrong Lower-Score Prediction",
        ),
        "",
        "## Interpretation of Correct Predictions",
        "",
        (
            "Correct predictions can show whether highly activated local "
            "regions are semantically consistent with the model's "
            "selected class."
        ),
        "",
        (
            "However, semantic consistency is evaluated after observing "
            "the model output. It does not prove that a highlighted "
            "phrase alone determined the classification."
        ),
        "",
        ("The strongest evidence supported by this pipeline is:"),
        "",
        (
            "> These local token regions produced strong internal "
            "convolutional responses in the trained TextCNN."
        ),
        "",
        "## Interpretation of Incorrect Predictions",
        "",
        ("Incorrect examples are essential to interpretation validation."),
        "",
        (
            "A strongly activated phrase accompanying a wrong prediction "
            "demonstrates that:"
        ),
        "",
        "- high activation does not imply a correct prediction;",
        "- locally informative text may still support the wrong class;",
        "- multiple emotional cues may compete;",
        ("- the classifier layer may combine filter activations imperfectly;"),
        ("- dataset label boundaries may overlap semantically."),
        "",
        (
            "Wrong high-score examples are especially important because "
            "they expose cases where the model is both confident and "
            "incorrect."
        ),
        "",
        "## What the Interpretation Supports",
        "",
        ("The current system supports the following claims:"),
        "",
        (
            "1. Exact high-activation convolutional windows can be "
            "traced back to input token spans."
        ),
        ("2. Different filters respond strongly to different local n-gram regions."),
        (
            "3. Activation evidence can be inspected for both correct "
            "and incorrect predictions."
        ),
        ("4. Padding-overlap maxima occur and must be handled explicitly."),
        (
            "5. Repeated filter/span behavior can be measured "
            "reproducibly from saved artifacts."
        ),
        "",
        "## What the Interpretation Does Not Support",
        "",
        "The current system does **not** support claims that:",
        "",
        "- highlighted text caused the predicted emotion;",
        "- activation magnitude is equivalent to class importance;",
        ("- a particular filter permanently represents one emotion;"),
        ("- the model has identified the writer's actual psychological state;"),
        "- high activation implies a prediction is correct;",
        (
            "- removing a highlighted phrase would necessarily change "
            "the prediction in a predictable way."
        ),
        "",
        (
            "Those questions would require additional techniques such "
            "as controlled input perturbation or class-specific "
            "attribution analysis."
        ),
        "",
        "## Current Interpretation Limitations",
        "",
        (
            f"- The aggregate validation sample contains only "
            f"{num_examples} held-out examples."
        ),
        (
            "- The sample is cohort-balanced rather than representative "
            "of the natural test distribution."
        ),
        (
            "- The method observes post-ReLU convolutional activations "
            "before the final classifier."
        ),
        (
            "- Filter activation strength is not directly equivalent to "
            "contribution to a particular output logit."
        ),
        "- Global max pooling discards non-maximal activations.",
        (
            "- Padding-overlap behavior exists because the original "
            "production model was not trained with explicit "
            "convolution-window masking."
        ),
        (
            "- Token-level spans are based on the project's simple "
            "tokenizer rather than subword tokenization."
        ),
        ("- Interpretation results are specific to `textcnn_dair_v001`."),
        "",
        "## Portfolio Significance",
        "",
        (
            "The interpretation layer changes TmotionLab from a system "
            "that only reports:"
        ),
        "",
        "```text",
        "text -> emotion prediction",
        "```",
        "",
        "into one that can also inspect:",
        "",
        "```text",
        "text",
        "-> prediction",
        "-> internal convolutional responses",
        "-> exact activated token regions",
        "-> correct/failure comparison",
        "```",
        "",
        (
            "This supports the broader TmotionLab thesis that model "
            "behavior should be inspectable rather than reduced to a "
            "leaderboard score."
        ),
        "",
        "## Next Step",
        "",
        "The next phase is **P8 — Error Lab**.",
        "",
        "The Error Lab will combine:",
        "",
        "- model predictions;",
        "- confusion pairs;",
        "- cross-model comparison cohorts;",
        "- high-score mistakes;",
        "- interpretation evidence;",
        "- manually reviewed error categories.",
        "",
        (
            "Interpretation metadata may support error analysis, but it "
            "should remain separate from analyst-authored hypotheses "
            "about why a prediction failed."
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

    print(f"[OK] Interpretation report: {args.output}")

    print(f"[RESULT] Examples documented: {num_examples}")

    print(
        f"[RESULT] Mean padding-max filters: {padding['mean_padding_max_filters']:.2f}"
    )


if __name__ == "__main__":
    main()
