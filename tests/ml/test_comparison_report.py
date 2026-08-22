from ml.scripts.generate_model_comparison_report import (
    markdown_class_table,
    markdown_confusion_table,
    pct,
)


def test_pct():
    assert pct(0.8265) == "82.65%"


def test_class_table():
    result = markdown_class_table(
        [
            {
                "true_label_name": "love",
                "count": 12,
            }
        ]
    )

    assert "love" in result
    assert "12" in result


def test_confusion_table():
    result = markdown_confusion_table(
        [
            {
                "true_label_name": "love",
                "baseline_prediction": "joy",
                "count": 7,
            }
        ],
        prediction_key="baseline_prediction",
    )

    assert "love" in result
    assert "joy" in result
    assert "7" in result
