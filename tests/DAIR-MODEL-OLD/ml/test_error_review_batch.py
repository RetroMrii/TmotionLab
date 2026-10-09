import pandas as pd
import pytest

from ml.scripts.build_error_review_batch import (
    BATCH_QUOTAS,
    build_review_batch,
)


def make_queue() -> pd.DataFrame:
    rows = []

    counter = 0

    for group, count in {
        "all_models_same_wrong_prediction": 25,
        "high_score_textcnn_disagreement": 15,
        "baseline_correct_textcnn_wrong": 15,
        "three_way_disagreement": 5,
        "other_textcnn_disagreement": 15,
    }.items():
        for _ in range(count):
            counter += 1

            rows.append(
                {
                    "review_queue_rank": counter,
                    "example_id": f"e{counter}",
                    "text": f"text {counter}",
                    "true_label_name": "joy",
                    "textcnn_prediction": "love",
                    "textcnn_score": 0.9,
                    "review_priority_group": group,
                    "baseline_prediction": "joy",
                    "numpy_prediction": "love",
                    "all_agree_and_wrong": False,
                    "baseline_correct_textcnn_wrong": False,
                    "three_way_disagreement": False,
                    "review_outcome": "",
                    "mechanism_tag": "",
                    "review_notes": "",
                    "reviewed_by": "",
                    "review_status": "unreviewed",
                }
            )

    return pd.DataFrame(rows)


def test_batch_has_50_examples():
    result = build_review_batch(make_queue())

    assert len(result) == 50


def test_batch_has_no_duplicates():
    result = build_review_batch(make_queue())

    assert not result["example_id"].duplicated().any()


def test_batch_respects_quotas():
    result = build_review_batch(make_queue())

    counts = result["review_priority_group"].value_counts().to_dict()

    for group, expected in BATCH_QUOTAS.items():
        assert counts[group] == expected


def test_missing_group_raises():
    frame = make_queue()

    frame = frame[frame["review_priority_group"] != "other_textcnn_disagreement"]

    with pytest.raises(ValueError):
        build_review_batch(frame)
