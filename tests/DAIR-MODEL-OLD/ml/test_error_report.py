import pandas as pd

from ml.scripts.generate_error_report import (
    confusion_table,
    example_table,
)


def test_confusion_table():
    frame = pd.DataFrame(
        [
            {
                "true_label_name": "love",
                "textcnn_prediction": "joy",
                "count": 12,
                "mean_textcnn_score": 0.81,
                "max_textcnn_score": 0.99,
                "high_score_errors": 7,
            }
        ]
    )

    result = confusion_table(frame)

    assert "love" in result
    assert "joy" in result
    assert "12" in result
    assert "0.8100" in result


def test_example_table():
    frame = pd.DataFrame(
        [
            {
                "example_id": "e1",
                "true_label_name": "sadness",
                "textcnn_prediction": "anger",
                "textcnn_score": 0.99,
                "baseline_prediction": "anger",
                "numpy_prediction": "anger",
                "text": "i feel frustrated",
            }
        ]
    )

    result = example_table(frame)

    assert "e1" in result
    assert "sadness" in result
    assert "anger" in result
    assert "i feel frustrated" in result
