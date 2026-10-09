import pandas as pd
import pytest

from ml.scripts.analyze_textcnn_interpretations import (
    build_filter_frequency,
    build_padding_summary,
    build_span_frequency,
    validate_record,
)


def make_record(
    *,
    example_id="e1",
    true_label="fear",
    prediction="fear",
):
    return {
        "model_id": "textcnn_test",
        "example_id": example_id,
        "cohort": "correct_high_score",
        "text": "i feel nervous",
        "true_label": true_label,
        "saved_prediction": prediction,
        "saved_top_score": 0.9,
        "tokens": [
            "i",
            "feel",
            "nervous",
        ],
        "interpretation": {
            "prediction": {
                "label": prediction,
                "score": 0.9,
            },
            "scores": [],
            "top_activations": [],
            "top_text_spans": [],
            "padding_activation_count": 3,
        },
    }


def test_validate_record_accepts_matching_prediction():
    validate_record(make_record())


def test_validate_record_rejects_prediction_mismatch():
    record = make_record()
    record["interpretation"]["prediction"]["label"] = "joy"

    with pytest.raises(ValueError):
        validate_record(record)


def test_filter_frequency_groups_filters():
    frame = pd.DataFrame(
        [
            {
                "example_id": "e1",
                "kernel_size": 3,
                "filter_index": 7,
                "predicted_label": "fear",
                "activation": 2.0,
                "rank": 1,
                "span_text": "feel nervous",
            },
            {
                "example_id": "e2",
                "kernel_size": 3,
                "filter_index": 7,
                "predicted_label": "fear",
                "activation": 3.0,
                "rank": 2,
                "span_text": "very nervous",
            },
        ]
    )

    result = build_filter_frequency(frame)

    assert result.iloc[0]["unique_examples"] == 2

    assert result.iloc[0]["occurrences"] == 2


def test_span_frequency_groups_text():
    frame = pd.DataFrame(
        [
            {
                "example_id": "e1",
                "span_text": "Very Nervous",
                "predicted_label": "fear",
                "activation": 2.0,
            },
            {
                "example_id": "e2",
                "span_text": "very nervous",
                "predicted_label": "fear",
                "activation": 3.0,
            },
        ]
    )

    result = build_span_frequency(frame)

    assert result.iloc[0]["span_text_normalized"] == "very nervous"

    assert result.iloc[0]["unique_examples"] == 2


def test_padding_summary():
    frame = build_padding_summary(
        [
            make_record(example_id="e1"),
            make_record(
                example_id="e2",
                true_label="joy",
                prediction="fear",
            ),
        ]
    )

    assert len(frame) == 2

    assert frame["padding_activation_count"].tolist() == [3, 3]
