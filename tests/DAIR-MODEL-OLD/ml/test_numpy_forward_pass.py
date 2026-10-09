import numpy as np

from ml.src.models.numpy_mlp import NumPyMLP


def test_forward_shapes():
    model = NumPyMLP(
        input_dim=10,
        hidden_dim=8,
        num_classes=3,
        seed=42,
    )

    X = np.zeros(
        (4, 10),
        dtype=np.float32,
    )

    probabilities, cache = model.forward(X)

    assert probabilities.shape == (4, 3)
    assert cache["z1"].shape == (4, 8)
    assert cache["h1"].shape == (4, 8)


def test_softmax_rows_sum_to_one():
    model = NumPyMLP(
        input_dim=5,
        hidden_dim=4,
        num_classes=3,
    )

    X = np.random.default_rng(42).normal(size=(6, 5)).astype(np.float32)

    probabilities = model.predict_proba(X)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(6),
        atol=1e-6,
    )


def test_cross_entropy_is_finite():
    probabilities = np.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float32,
    )

    labels = np.array(
        [0, 1],
        dtype=np.int64,
    )

    loss = NumPyMLP.cross_entropy_loss(
        probabilities,
        labels,
    )

    assert np.isfinite(loss)
    assert loss > 0


def test_model_save_and_load(tmp_path):
    model = NumPyMLP(
        input_dim=6,
        hidden_dim=4,
        num_classes=3,
        seed=42,
    )

    X = np.random.default_rng(7).normal(size=(3, 6)).astype(np.float32)

    before = model.predict_proba(X)

    path = tmp_path / "model.npz"

    model.save(str(path))

    loaded = NumPyMLP.load(str(path))

    after = loaded.predict_proba(X)

    np.testing.assert_allclose(
        before,
        after,
        atol=1e-7,
    )
