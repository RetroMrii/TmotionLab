import numpy as np

from ml.src.models.numpy_mlp import NumPyMLP


def test_backward_gradient_shapes():
    model = NumPyMLP(
        input_dim=7,
        hidden_dim=5,
        num_classes=3,
        seed=42,
    )

    X = np.random.default_rng(1).normal(size=(4, 7)).astype(np.float32)

    labels = np.array(
        [0, 1, 2, 1],
        dtype=np.int64,
    )

    _, cache = model.forward(X)

    gradients = model.backward(
        cache,
        labels,
    )

    assert gradients["W1"].shape == model.W1.shape
    assert gradients["b1"].shape == model.b1.shape
    assert gradients["W2"].shape == model.W2.shape
    assert gradients["b2"].shape == model.b2.shape


def test_single_training_batch_reduces_loss():
    model = NumPyMLP(
        input_dim=4,
        hidden_dim=8,
        num_classes=2,
        seed=42,
    )

    X = np.array(
        [
            [1, 0, 0, 0],
            [1, 0.1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0.1],
        ],
        dtype=np.float32,
    )

    labels = np.array(
        [0, 0, 1, 1],
        dtype=np.int64,
    )

    initial_probabilities = model.predict_proba(X)

    initial_loss = model.cross_entropy_loss(
        initial_probabilities,
        labels,
    )

    for _ in range(100):
        model.train_batch(
            X,
            labels,
            learning_rate=0.1,
        )

    final_probabilities = model.predict_proba(X)

    final_loss = model.cross_entropy_loss(
        final_probabilities,
        labels,
    )

    assert final_loss < initial_loss
