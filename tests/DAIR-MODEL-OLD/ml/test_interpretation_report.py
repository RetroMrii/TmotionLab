import pandas as pd

from ml.scripts.generate_interpretation_report import (
    cohort_padding_table,
    markdown_filter_table,
    markdown_span_table,
)


def test_filter_table():
    frame = pd.DataFrame(
        [
            {
                "kernel_size": 3,
                "filter_index": 7,
                "predicted_label": "fear",
                "unique_examples": 4,
                "mean_activation": 3.25,
                "mean_rank": 2.5,
            }
        ]
    )

    result = markdown_filter_table(frame)

    assert "fear" in result
    assert "3.2500" in result


def test_span_table():
    frame = pd.DataFrame(
        [
            {
                "span_text_normalized": ("very nervous"),
                "predicted_label": "fear",
                "unique_examples": 3,
                "mean_activation": 2.75,
            }
        ]
    )

    result = markdown_span_table(frame)

    assert "very nervous" in result
    assert "2.7500" in result


def test_padding_table():
    frame = pd.DataFrame(
        {
            "cohort": [
                "correct_high_score",
                "correct_high_score",
                "wrong_high_score",
            ],
            "padding_activation_count": [
                10,
                20,
                30,
            ],
        }
    )

    result = cohort_padding_table(frame)

    assert "correct_high_score" in result
    assert "15.00" in result
    assert "wrong_high_score" in result
