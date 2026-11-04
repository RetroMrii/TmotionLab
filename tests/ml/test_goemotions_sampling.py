import pytest
import torch

from ml.src.data.goemotions_sampling import (
    build_sampling_weights,
    sample_epoch_indices,
    summarize_sampling,
)


LABELS = [
    "grief",
    "nervousness",
    "pride",
    "relief",
    "joy",
    "sadness",
]

ROWS = [
    {"labels": [0, 5]},
    {"labels": [1]},
    {"labels": [2, 4]},
    {"labels": [3]},
    {"labels": [4]},
    {"labels": [4, 5]},
]


def test_uniform_weights():
    weights = build_sampling_weights(ROWS, LABELS, "uniform")

    assert weights.dtype == torch.float64
    assert weights.tolist() == [1.0] * 6


def test_rare_label_weights():
    weights = build_sampling_weights(ROWS, LABELS, "rare_oversample")

    assert weights.tolist() == [4.0, 4.0, 4.0, 4.0, 1.0, 1.0]


def test_reproducible_sampling():
    weights = torch.tensor(
        [4.0, 1.0, 1.0],
        dtype=torch.float64,
    )

    first = sample_epoch_indices(weights, 42, 0)
    second = sample_epoch_indices(weights, 42, 0)

    assert torch.equal(first, second)
    assert len(first) == len(weights)
    assert first.min() >= 0
    assert first.max() < len(weights)


def test_sampling_summary():
    weights = build_sampling_weights(ROWS, LABELS, "rare_oversample")

    indices = torch.tensor([0, 0, 1, 2, 3, 4])

    report = summarize_sampling(ROWS, LABELS, weights, indices)

    assert report["training_examples"] == 6
    assert report["epoch_draws"] == 6
    assert report["unique_examples_sampled"] == 5

    assert report["target_labels"]["grief"]["sampled_positive_occurrences"] == 2


def test_invalid_strategy():
    with pytest.raises(ValueError):
        build_sampling_weights(ROWS, LABELS, "unknown")
