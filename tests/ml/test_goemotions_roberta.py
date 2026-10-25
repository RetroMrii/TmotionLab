import numpy as np
import pytest
import torch

from ml.scripts.train_goemotions_roberta import (
    sigmoid,
    make_targets,
    compute_metrics,
    validate_config,
)


def test_sigmoid_probabilities():
    logits = np.array(
        [
            [0.0, 2.0, -2.0],
            [1000.0, -1000.0, 0.0],
        ]
    )

    probabilities = sigmoid(logits)

    assert probabilities.shape == logits.shape
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert probabilities[0, 0] == pytest.approx(0.5)
    assert probabilities[1, 0] == pytest.approx(1.0)
    assert probabilities[1, 1] == pytest.approx(0.0)


def test_target_encoding():
    rows = [
        {"labels": [1, 3]},
        {"labels": [0]},
    ]

    result = make_targets(rows, 4)

    assert result.tolist() == [
        [0, 1, 0, 1],
        [1, 0, 0, 0],
    ]


def test_macro_average_precision():
    logits = np.array(
        [
            [5.0, -5.0],
            [-5.0, 5.0],
            [5.0, 5.0],
        ]
    )

    labels = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )

    result = compute_metrics(
        type(
            "EvalPrediction",
            (),
            {"predictions": logits, "label_ids": labels},
        )()
    )

    assert result["macro_average_precision"] == pytest.approx(1.0)


def test_multilabel_bce_loss():
    logits = torch.tensor(
        [
            [1.0, -1.0],
            [-0.5, 2.0],
        ]
    )

    labels = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    loss = torch.nn.BCEWithLogitsLoss()(logits, labels)

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_configuration_validation():
    config = {
        "experiment_id": "roberta_goemotions_multilabel_real_v001",
        "dataset_version": "goemotions_v001",
        "num_labels": 28,
        "max_length": 48,
        "checkpoint_selection": {"metric": "macro_average_precision"},
        "threshold_candidates": [0.1, 0.2, 0.5],
    }

    manifest = {
        "dataset_version": "goemotions_v001",
        "label_names": [str(i) for i in range(28)],
    }

    validate_config(config, manifest)

    config["max_length"] = 600

    with pytest.raises(ValueError):
        validate_config(config, manifest)
