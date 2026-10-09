from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_ERRORS = Path("ml/artifacts/errors/textcnn_dair_v001_errors.csv")

DEFAULT_OUTPUT_DIR = Path("ml/artifacts/errors")


REVIEW_OUTCOMES = [
    "clear_model_error",
    "plausible_model_prediction",
    "ambiguous_label_boundary",
    "possible_annotation_mismatch",
    "insufficient_context",
]


MECHANISM_TAGS = [
    "semantic_overlap",
    "negation",
    "sarcasm_or_irony",
    "rare_label",
    "mixed_emotion",
    "preprocessing_artifact",
    "lexical_bias",
    "other",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the prioritized human-review queue for TmotionLab Error Lab."
        )
    )

    parser.add_argument(
        "--errors",
        type=Path,
        default=DEFAULT_ERRORS,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def load_errors(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Error dataset not found: {path}")

    frame = pd.read_csv(path)

    required = {
        "error_rank",
        "example_id",
        "text",
        "true_label_name",
        "textcnn_prediction",
        "textcnn_score",
        "confusion_pair",
        "high_score_error",
        "baseline_prediction",
        "baseline_correct",
        "numpy_prediction",
        "numpy_correct",
        "all_models_agree",
        "all_models_wrong",
        "all_agree_and_wrong",
        "baseline_correct_textcnn_wrong",
        "three_way_disagreement",
        "interpretation_available",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"Error dataset missing required columns: {sorted(missing)}")

    if frame["example_id"].duplicated().any():
        raise ValueError("Error dataset contains duplicate example_id values")

    return frame


def priority_group(
    row: pd.Series,
) -> str:
    if bool(row["all_agree_and_wrong"]):
        return "all_models_same_wrong_prediction"

    if bool(row["high_score_error"]):
        return "high_score_textcnn_disagreement"

    if bool(row["baseline_correct_textcnn_wrong"]):
        return "baseline_correct_textcnn_wrong"

    if bool(row["three_way_disagreement"]):
        return "three_way_disagreement"

    return "other_textcnn_disagreement"


def priority_group_rank(
    group: str,
) -> int:
    ranking = {
        "all_models_same_wrong_prediction": 1,
        "high_score_textcnn_disagreement": 2,
        "baseline_correct_textcnn_wrong": 3,
        "three_way_disagreement": 4,
        "other_textcnn_disagreement": 5,
    }

    return ranking[group]


def build_review_queue(
    errors: pd.DataFrame,
) -> pd.DataFrame:
    queue = errors.copy()

    queue["review_priority_group"] = queue.apply(
        priority_group,
        axis=1,
    )

    queue["review_priority_group_rank"] = queue["review_priority_group"].map(
        priority_group_rank
    )

    queue["cross_model_consensus"] = (
        queue["baseline_prediction"].astype(str)
        + " | "
        + queue["numpy_prediction"].astype(str)
        + " | "
        + queue["textcnn_prediction"].astype(str)
    )

    queue["all_three_predict_same_label"] = (
        queue["baseline_prediction"] == queue["numpy_prediction"]
    ) & (queue["numpy_prediction"] == queue["textcnn_prediction"])

    # Human-review fields.
    queue["review_outcome"] = ""
    queue["mechanism_tag"] = ""
    queue["review_notes"] = ""
    queue["reviewed_by"] = ""
    queue["review_status"] = "unreviewed"

    queue = queue.sort_values(
        [
            "review_priority_group_rank",
            "textcnn_score",
            "error_rank",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    ).reset_index(drop=True)

    queue.insert(
        0,
        "review_queue_rank",
        range(
            1,
            len(queue) + 1,
        ),
    )

    return queue


def build_schema() -> dict:
    return {
        "schema_version": "1.0",
        "model_id": "textcnn_dair_v001",
        "purpose": (
            "Human review of disagreements between "
            "TextCNN predictions and benchmark labels."
        ),
        "important_rule": (
            "The benchmark label remains unchanged. "
            "Review fields record analyst assessment "
            "and do not overwrite ground truth."
        ),
        "review_outcomes": {
            "clear_model_error": (
                "The benchmark label is substantially "
                "better supported by the available text "
                "than the model prediction."
            ),
            "plausible_model_prediction": (
                "The model prediction is semantically "
                "reasonable, although the benchmark label "
                "is also defensible."
            ),
            "ambiguous_label_boundary": (
                "Two or more emotion labels are strongly "
                "defensible from the available text."
            ),
            "possible_annotation_mismatch": (
                "The available text appears to support "
                "the model prediction substantially more "
                "strongly than the benchmark label."
            ),
            "insufficient_context": (
                "The sentence does not contain enough "
                "context to confidently adjudicate the "
                "benchmark/model disagreement."
            ),
        },
        "mechanism_tags": {
            "semantic_overlap": ("Emotion classes overlap semantically."),
            "negation": (
                "Negation may materially affect the interpretation of the sentence."
            ),
            "sarcasm_or_irony": (
                "Literal wording may not reflect the intended emotional meaning."
            ),
            "rare_label": (
                "The disagreement involves a relatively rare emotion class."
            ),
            "mixed_emotion": ("The text contains evidence for multiple emotions."),
            "preprocessing_artifact": (
                "Tokenization or preprocessing may obscure important information."
            ),
            "lexical_bias": (
                "One or more strong lexical cues may dominate the model decision."
            ),
            "other": ("A mechanism not covered by another tag."),
        },
        "review_status_values": [
            "unreviewed",
            "reviewed",
            "needs_second_review",
        ],
    }


def build_review_summary(
    queue: pd.DataFrame,
) -> dict:
    group_counts = queue["review_priority_group"].value_counts().to_dict()

    same_prediction_count = int(queue["all_three_predict_same_label"].sum())

    interpretation_count = int(queue["interpretation_available"].sum())

    return {
        "model_id": "textcnn_dair_v001",
        "num_disagreements": int(len(queue)),
        "num_unreviewed": int((queue["review_status"] == "unreviewed").sum()),
        "all_three_same_prediction_count": (same_prediction_count),
        "interpretation_available_count": (interpretation_count),
        "priority_group_counts": (group_counts),
        "review_outcomes": (REVIEW_OUTCOMES),
        "mechanism_tags": (MECHANISM_TAGS),
    }


def main() -> None:
    args = parse_args()

    errors = load_errors(args.errors)

    queue = build_review_queue(errors)

    schema = build_schema()

    summary = build_review_summary(queue)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    queue_path = args.output_dir / "textcnn_dair_v001_review_queue.csv"

    schema_path = args.output_dir / "textcnn_dair_v001_review_schema.json"

    summary_path = args.output_dir / "textcnn_dair_v001_review_summary.json"

    queue.to_csv(
        queue_path,
        index=False,
        encoding="utf-8",
    )

    with schema_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            schema,
            handle,
            indent=2,
            ensure_ascii=False,
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

    print(f"[OK] Benchmark disagreements loaded: {len(errors)}")

    print(f"[RESULT] Review queue size: {len(queue)}")

    for group, count in (
        queue["review_priority_group"].value_counts().sort_index().items()
    ):
        print(f"[RESULT] {group}: {count}")

    print(
        "[RESULT] All three predict same "
        "alternative label: "
        f"{summary['all_three_same_prediction_count']}"
    )

    print(
        "[RESULT] Interpretation available: "
        f"{summary['interpretation_available_count']}"
    )

    print(f"[OK] Review queue: {queue_path}")

    print(f"[OK] Review schema: {schema_path}")

    print(f"[OK] Review summary: {summary_path}")


if __name__ == "__main__":
    main()
