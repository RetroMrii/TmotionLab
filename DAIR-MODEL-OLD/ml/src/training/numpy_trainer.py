from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score

from ml.src.models.numpy_mlp import NumPyMLP


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_weighted_f1: float


def iter_minibatches(
    X: sparse.spmatrix,
    y: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
):
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if X.shape[0] != len(y):
        raise ValueError("X and y must contain the same number of rows")

    indices = np.arange(X.shape[0])
    rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]

        X_batch = X[batch_indices].toarray().astype(np.float32, copy=False)

        y_batch = y[batch_indices]

        yield X_batch, y_batch


def predict_sparse(
    model: NumPyMLP,
    X: sparse.spmatrix,
    *,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    all_predictions = []
    all_probabilities = []

    for start in range(0, X.shape[0], batch_size):
        X_batch = X[start : start + batch_size].toarray().astype(np.float32, copy=False)

        probabilities = model.predict_proba(X_batch)
        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        all_probabilities.append(probabilities)
        all_predictions.append(predictions)

    return (
        np.concatenate(all_predictions),
        np.concatenate(all_probabilities),
    )


def evaluate_sparse(
    model: NumPyMLP,
    X: sparse.spmatrix,
    y: np.ndarray,
    *,
    batch_size: int = 512,
) -> dict:
    predictions, probabilities = predict_sparse(
        model,
        X,
        batch_size=batch_size,
    )

    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
            )
        ),
        "weighted_f1": float(
            f1_score(
                y,
                predictions,
                average="weighted",
            )
        ),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def train_epoch(
    model: NumPyMLP,
    X_train: sparse.spmatrix,
    y_train: np.ndarray,
    *,
    batch_size: int,
    learning_rate: float,
    rng: np.random.Generator,
) -> float:
    total_loss = 0.0
    total_examples = 0

    for X_batch, y_batch in iter_minibatches(
        X_train,
        y_train,
        batch_size=batch_size,
        rng=rng,
    ):
        loss = model.train_batch(
            X_batch,
            y_batch,
            learning_rate=learning_rate,
        )

        batch_examples = len(y_batch)

        total_loss += loss * batch_examples
        total_examples += batch_examples

    return total_loss / total_examples
