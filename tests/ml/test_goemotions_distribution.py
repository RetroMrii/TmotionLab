import pytest

from ml.scripts.analyze_goemotions_label_distribution import (
    analyze_distribution,
    calculate_budget,
)


def test_distribution_counts_multilabel_examples():
    rows = [
        {"labels": [0, 1]},
        {"labels": [0]},
        {"labels": [2]},
    ]

    summary = analyze_distribution(
        rows,
        ["grief", "pride", "joy"],
    )

    by_label = {item["label"]: item for item in summary["distribution"]}

    assert summary["training_examples"] == 3
    assert summary["total_positive_label_occurrences"] == 4
    assert summary["mean_labels_per_example"] == pytest.approx(4 / 3)

    assert by_label["grief"]["positive_examples"] == 2
    assert by_label["pride"]["positive_examples"] == 1
    assert by_label["joy"]["positive_examples"] == 1


def test_budget_calculation():
    distribution = [
        {"label": "grief", "positive_examples": 10},
        {"label": "pride", "positive_examples": 20},
        {"label": "joy", "positive_examples": 100},
    ]

    budget = calculate_budget(
        distribution,
        target_labels=("grief", "pride"),
        multiplier=3,
    )

    assert budget["largest_label_count"] == 100
    assert budget["moderate_total_additions"] == 90

    assert budget["target_budgets"]["grief"]["moderate_synthetic_target"] == 30

    assert budget["target_budgets"]["pride"]["additions_to_match_largest_label"] == 80

    assert budget["full_28_label_equalization_estimate"] == 170


def test_invalid_duplicate_labels():
    with pytest.raises(ValueError):
        analyze_distribution(
            [{"labels": [0, 0]}],
            ["grief"],
        )


def test_missing_budget_target():
    with pytest.raises(ValueError):
        calculate_budget(
            [{"label": "joy", "positive_examples": 100}],
            target_labels=("grief",),
        )
