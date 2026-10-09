import pandas as pd

from ml.scripts.analyze_model_behavior import (
    class_counts,
    confusion_counts,
    rate,
)


def test_rate():
    assert rate(5, 10) == 0.5


def test_rate_zero_denominator():
    assert rate(5, 0) == 0.0


def test_class_counts():
    frame = pd.DataFrame(
        {
            "true_label_name": [
                "joy",
                "joy",
                "fear",
            ]
        }
    )

    result = class_counts(frame)

    counts = dict(
        zip(
            result["true_label_name"],
            result["count"],
        )
    )

    assert counts["joy"] == 2
    assert counts["fear"] == 1


def test_confusion_counts():
    frame = pd.DataFrame(
        {
            "true_label_name": [
                "love",
                "love",
                "fear",
            ],
            "baseline_prediction": [
                "joy",
                "joy",
                "sadness",
            ],
        }
    )

    result = confusion_counts(
        frame,
        prediction_column=("baseline_prediction"),
    )

    assert result[0]["true_label_name"] == "love"
    assert result[0]["baseline_prediction"] == "joy"
    assert result[0]["count"] == 2
