import numpy as np
from scipy import sparse

from ml.src.models.numpy_mlp import NumPyMLP
from ml.src.training.numpy_trainer import (
    evaluate_sparse,
    iter_minibatches,
    train_epoch,
)


def test_minibatches_cover_all_examples():
    X = sparse.csr_matrix(
        np.arange(
            30,
            dtype=np.float32,
        ).reshape(10, 3)
    )

    y = np.arange(10)

    rng = np.random.default_rng(42)

    batches = list(
        iter_minibatches(
            X,
            y,
            batch_size=4,
            rng=rng,
        )
    )

    total_examples = sum(len(y_batch) for _, y_batch in batches)

    assert total_examples == 10


def test_train_epoch_returns_finite_loss():
    X = sparse.csr_matrix(
        np.array(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
            ],
            dtype=np.float32,
        )
    )

    y = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    model = NumPyMLP(
        input_dim=4,
        hidden_dim=8,
        num_classes=2,
        seed=42,
    )

    rng = np.random.default_rng(42)

    loss = train_epoch(
        model,
        X,
        y,
        batch_size=2,
        learning_rate=0.1,
        rng=rng,
    )

    assert np.isfinite(loss)
    assert loss > 0


def test_evaluate_sparse_shapes():
    X = sparse.csr_matrix(
        np.eye(
            6,
            dtype=np.float32,
        )
    )

    y = np.array(
        [0, 1, 2, 0, 1, 2],
        dtype=np.int64,
    )

    model = NumPyMLP(
        input_dim=6,
        hidden_dim=5,
        num_classes=3,
        seed=42,
    )

    result = evaluate_sparse(
        model,
        X,
        y,
        batch_size=2,
    )

    assert result["predictions"].shape == (6,)
    assert result["probabilities"].shape == (
        6,
        3,
    )

    assert 0 <= result["accuracy"] <= 1
    assert 0 <= result["macro_f1"] <= 1
