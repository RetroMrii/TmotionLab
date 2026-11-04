import numpy as np
import pytest

from ml.scripts.diagnose_goemotions_rare_labels import (
    score_summary,
    threshold_analysis,
    analyze_label,
)


def test_score_summary():
    summary = score_summary([0.1, 0.2, 0.3])

    assert summary["median"] == pytest.approx(0.2)
    assert summary["max"] == pytest.approx(0.3)
    assert score_summary([]) is None


def test_threshold_sweep_recovers_low_scores():
    truth = np.array([1, 1, 0, 0], dtype=bool)
    scores = np.array([0.12, 0.09, 0.02, 0.01])

    best, results = threshold_analysis(
        truth,
        scores,
        np.array([0.05, 0.10, 0.25]),
    )

    assert best["threshold"] == pytest.approx(0.05)
    assert best["f1"] == pytest.approx(1.0)
    assert len(results) == 3


def test_label_diagnostics():
    names = ["rare", "common"]

    truth = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 1],
            [0, 1],
        ],
        dtype=np.int8,
    )

    probabilities = np.array(
        [
            [0.12, 0.10],
            [0.09, 0.85],
            [0.02, 0.80],
            [0.01, 0.90],
        ]
    )

    result = analyze_label(
        0,
        names,
        truth,
        probabilities,
        global_threshold=0.25,
    )

    assert result["support"] == 2
    assert result["global_predicted_positive"] == 0
    assert result["average_precision"] == pytest.approx(1.0)
    assert result["best_diagnostic_threshold"]["f1"] == (pytest.approx(1.0))
    assert result["top_alternative_predictions"][0] == {
        "label": "common",
        "count": 1,
    }
