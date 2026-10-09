import pandas as pd
import pytest

from ml.scripts.compare_models import (
    build_comparison,
    build_summary,
    validate_alignment,
)


def make_frame(
    predictions,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": [
                "e1",
                "e2",
                "e3",
            ],
            "text": [
                "text one",
                "text two",
                "text three",
            ],
            "true_label": [
                0,
                1,
                2,
            ],
            "true_label_name": [
                "sadness",
                "joy",
                "love",
            ],
            "predicted_label": predictions,
            "predicted_label_name": [
                [
                    "sadness",
                    "joy",
                    "love",
                ][value]
                for value in predictions
            ],
            "top_score": [
                0.8,
                0.7,
                0.6,
            ],
        }
    )


def test_alignment_accepts_identical_examples():
    first = make_frame([0, 1, 2])
    second = make_frame([0, 2, 2])

    validate_alignment(
        first,
        second,
        candidate_name="test",
    )


def test_alignment_rejects_different_ids():
    first = make_frame([0, 1, 2])
    second = make_frame([0, 1, 2])

    second.loc[1, "example_id"] = "wrong"

    with pytest.raises(ValueError):
        validate_alignment(
            first,
            second,
            candidate_name="test",
        )


def test_comparison_flags():
    baseline = make_frame([0, 0, 1])

    numpy_model = make_frame([0, 1, 1])

    textcnn = make_frame([0, 1, 2])

    result = build_comparison(
        baseline,
        numpy_model,
        textcnn,
    )

    assert result.loc[
        0,
        "all_agree_and_correct",
    ]

    assert result.loc[
        1,
        "baseline_wrong_textcnn_correct",
    ]

    assert result.loc[
        2,
        "baseline_wrong_textcnn_correct",
    ]

    assert result.loc[
        2,
        "numpy_wrong_textcnn_correct",
    ]


def test_summary_counts_examples():
    baseline = make_frame([0, 0, 1])

    numpy_model = make_frame([0, 1, 1])

    textcnn = make_frame([0, 1, 2])

    comparison = build_comparison(
        baseline,
        numpy_model,
        textcnn,
    )

    summary = build_summary(comparison)

    assert summary["num_examples"] == 3

    assert summary["counts"]["baseline_wrong_textcnn_correct"] == 2
