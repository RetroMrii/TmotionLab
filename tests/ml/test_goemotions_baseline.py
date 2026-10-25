import numpy as np
import pytest
from sklearn.metrics import f1_score

from ml.scripts.train_goemotions_tfidf import (
    make_targets,
    choose_threshold,
    evaluate,
)


def test_multihot_targets():
    rows = [
        {"labels": [0, 2]},
        {"labels": [1]},
        {"labels": [0, 1, 2]},
    ]

    result = make_targets(rows, 3)

    assert result.tolist() == [
        [1, 0, 1],
        [0, 1, 0],
        [1, 1, 1],
    ]


def test_threshold_selection_maximizes_macro_f1():
    truth = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )

    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.3, 0.7],
            [0.7, 0.6],
        ]
    )

    candidates = [0.3, 0.5, 0.8]

    threshold, results = choose_threshold(truth, probabilities, candidates)

    best_score = max(row["macro_f1"] for row in results)
    actual_score = f1_score(
        truth,
        (probabilities >= threshold).astype(int),
        average="macro",
        zero_division=0,
    )

    assert actual_score == pytest.approx(best_score)


def test_evaluation_reports_multilabel_metrics():
    truth = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )

    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.9, 0.9],
        ]
    )

    metrics = evaluate(
        truth,
        probabilities,
        threshold=0.5,
        label_names=["a", "b"],
    )

    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["micro_f1"] == pytest.approx(1.0)
    assert set(metrics["per_label"]) == {"a", "b"}
    assert metrics["per_label"]["a"]["support"] == 2
