import pandas as pd

from ml.scripts.build_error_review_queue import (
    build_review_queue,
    build_review_summary,
    build_schema,
)


def make_errors() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "error_rank": [
                1,
                2,
                3,
                4,
                5,
            ],
            "example_id": [
                "e1",
                "e2",
                "e3",
                "e4",
                "e5",
            ],
            "text": [
                "one",
                "two",
                "three",
                "four",
                "five",
            ],
            "true_label_name": [
                "sadness",
                "joy",
                "love",
                "fear",
                "anger",
            ],
            "textcnn_prediction": [
                "anger",
                "love",
                "joy",
                "surprise",
                "sadness",
            ],
            "textcnn_score": [
                0.99,
                0.95,
                0.70,
                0.60,
                0.55,
            ],
            "confusion_pair": [
                "sadness -> anger",
                "joy -> love",
                "love -> joy",
                "fear -> surprise",
                "anger -> sadness",
            ],
            "high_score_error": [
                True,
                True,
                False,
                False,
                False,
            ],
            "baseline_prediction": [
                "anger",
                "joy",
                "love",
                "joy",
                "fear",
            ],
            "baseline_correct": [
                False,
                True,
                True,
                False,
                False,
            ],
            "numpy_prediction": [
                "anger",
                "love",
                "joy",
                "anger",
                "sadness",
            ],
            "numpy_correct": [
                False,
                False,
                False,
                False,
                False,
            ],
            "all_models_agree": [
                True,
                False,
                False,
                False,
                False,
            ],
            "all_models_wrong": [
                True,
                False,
                False,
                True,
                True,
            ],
            "all_agree_and_wrong": [
                True,
                False,
                False,
                False,
                False,
            ],
            "baseline_correct_textcnn_wrong": [
                False,
                True,
                True,
                False,
                False,
            ],
            "three_way_disagreement": [
                False,
                False,
                False,
                True,
                False,
            ],
            "interpretation_available": [
                True,
                True,
                False,
                False,
                False,
            ],
        }
    )


def test_priority_order():
    queue = build_review_queue(make_errors())

    assert queue.iloc[0]["review_priority_group"] == "all_models_same_wrong_prediction"


def test_high_score_is_second_priority():
    queue = build_review_queue(make_errors())

    row = queue[queue["example_id"] == "e2"].iloc[0]

    assert row["review_priority_group"] == "high_score_textcnn_disagreement"


def test_baseline_regression_group():
    queue = build_review_queue(make_errors())

    row = queue[queue["example_id"] == "e3"].iloc[0]

    assert row["review_priority_group"] == "baseline_correct_textcnn_wrong"


def test_three_way_group():
    queue = build_review_queue(make_errors())

    row = queue[queue["example_id"] == "e4"].iloc[0]

    assert row["review_priority_group"] == "three_way_disagreement"


def test_review_fields_start_blank():
    queue = build_review_queue(make_errors())

    assert (queue["review_outcome"] == "").all()

    assert (queue["mechanism_tag"] == "").all()

    assert (queue["review_status"] == "unreviewed").all()


def test_schema_contains_review_outcomes():
    schema = build_schema()

    assert "possible_annotation_mismatch" in schema["review_outcomes"]

    assert "ambiguous_label_boundary" in schema["review_outcomes"]


def test_summary_counts_queue():
    queue = build_review_queue(make_errors())

    summary = build_review_summary(queue)

    assert summary["num_disagreements"] == 5

    assert summary["num_unreviewed"] == 5
