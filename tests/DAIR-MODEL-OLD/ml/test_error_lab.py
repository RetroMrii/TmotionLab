import pandas as pd
import pytest

from ml.scripts.build_error_lab import (
    build_confusion_summary,
    build_error_lab_dataset,
    extract_interpretation_fields,
)


def make_predictions() -> pd.DataFrame:
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
            "predicted_label": [
                0,
                0,
                1,
            ],
            "predicted_label_name": [
                "sadness",
                "sadness",
                "joy",
            ],
            "top_score": [
                0.9,
                0.85,
                0.7,
            ],
        }
    )


def make_comparison() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": [
                "e1",
                "e2",
                "e3",
            ],
            "baseline_prediction": [
                "sadness",
                "joy",
                "love",
            ],
            "baseline_score": [
                0.8,
                0.7,
                0.6,
            ],
            "baseline_correct": [
                True,
                True,
                True,
            ],
            "numpy_prediction": [
                "sadness",
                "sadness",
                "joy",
            ],
            "numpy_score": [
                0.8,
                0.8,
                0.7,
            ],
            "numpy_correct": [
                True,
                False,
                False,
            ],
            "textcnn_prediction": [
                "sadness",
                "sadness",
                "joy",
            ],
            "textcnn_score": [
                0.9,
                0.85,
                0.7,
            ],
            "textcnn_correct": [
                True,
                False,
                False,
            ],
            "all_models_agree": [
                True,
                False,
                False,
            ],
            "all_models_correct": [
                True,
                False,
                False,
            ],
            "all_models_wrong": [
                False,
                False,
                False,
            ],
            "all_agree_and_wrong": [
                False,
                False,
                False,
            ],
            "baseline_wrong_textcnn_correct": [
                False,
                False,
                False,
            ],
            "baseline_correct_textcnn_wrong": [
                False,
                True,
                True,
            ],
            "numpy_wrong_textcnn_correct": [
                False,
                False,
                False,
            ],
            "numpy_correct_textcnn_wrong": [
                False,
                False,
                False,
            ],
            "three_way_disagreement": [
                False,
                False,
                False,
            ],
        }
    )


def test_build_error_dataset_keeps_only_errors():
    result = build_error_lab_dataset(
        make_predictions(),
        make_comparison(),
        {},
        high_score_threshold=0.8,
    )

    assert len(result) == 2

    assert set(result["example_id"]) == {
        "e2",
        "e3",
    }


def test_high_score_error_threshold():
    result = build_error_lab_dataset(
        make_predictions(),
        make_comparison(),
        {},
        high_score_threshold=0.8,
    )

    values = dict(
        zip(
            result["example_id"],
            result["high_score_error"],
        )
    )

    assert bool(values["e2"]) is True
    assert bool(values["e3"]) is False


def test_confusion_pair_created():
    result = build_error_lab_dataset(
        make_predictions(),
        make_comparison(),
        {},
        high_score_threshold=0.8,
    )

    pairs = set(result["confusion_pair"])

    assert "joy -> sadness" in pairs
    assert "love -> joy" in pairs


def test_manual_fields_are_unreviewed():
    result = build_error_lab_dataset(
        make_predictions(),
        make_comparison(),
        {},
        high_score_threshold=0.8,
    )

    assert (result["review_status"] == "unreviewed").all()

    assert (result["error_category"] == "").all()


def test_interpretation_fields_missing():
    result = extract_interpretation_fields(None)

    assert result["interpretation_available"] is False

    assert result["top_activation_span_1"] is None


def test_interpretation_fields_present():
    record = {
        "cohort": "wrong_high_score",
        "interpretation": {
            "padding_activation_count": 12,
            "top_text_spans": [{"span_text": ("feel very nervous")}],
        },
    }

    result = extract_interpretation_fields(record)

    assert result["interpretation_available"] is True

    assert result["padding_activation_count"] == 12

    assert result["top_activation_span_1"] == "feel very nervous"


def test_confusion_summary_counts_pairs():
    errors = build_error_lab_dataset(
        make_predictions(),
        make_comparison(),
        {},
        high_score_threshold=0.8,
    )

    result = build_confusion_summary(errors)

    assert result["count"].sum() == 2


def test_invalid_threshold_rejected():
    with pytest.raises(ValueError):
        build_error_lab_dataset(
            make_predictions(),
            make_comparison(),
            {},
            high_score_threshold=1.5,
        )
