from __future__ import annotations

import numpy as np


class NumPyMLP:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        seed: int = 42,
    ) -> None:
        if input_dim < 1:
            raise ValueError("input_dim must be >= 1")

        if hidden_dim < 1:
            raise ValueError("hidden_dim must be >= 1")

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        rng = np.random.default_rng(seed)

        self.W1 = rng.standard_normal((input_dim, hidden_dim)).astype(
            np.float32
        ) * np.sqrt(2.0 / input_dim)

        self.b1 = np.zeros(
            (1, hidden_dim),
            dtype=np.float32,
        )

        self.W2 = rng.standard_normal((hidden_dim, num_classes)).astype(
            np.float32
        ) * np.sqrt(2.0 / hidden_dim)

        self.b2 = np.zeros(
            (1, num_classes),
            dtype=np.float32,
        )

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(
            logits,
            axis=1,
            keepdims=True,
        )

        exp_values = np.exp(shifted)

        return exp_values / np.sum(
            exp_values,
            axis=1,
            keepdims=True,
        )

    @staticmethod
    def cross_entropy_loss(
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        if probabilities.ndim != 2:
            raise ValueError("probabilities must be a 2D array")

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")

        if len(probabilities) != len(labels):
            raise ValueError(
                "probabilities and labels must contain the same number of examples"
            )

        epsilon = 1e-12

        true_probabilities = probabilities[
            np.arange(len(labels)),
            labels,
        ]

        return float(
            -np.mean(
                np.log(
                    np.clip(
                        true_probabilities,
                        epsilon,
                        1.0,
                    )
                )
            )
        )

    def forward(
        self,
        X: np.ndarray,
    ) -> tuple[
        np.ndarray,
        dict[str, np.ndarray],
    ]:
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {X.shape[1]}")

        z1 = X @ self.W1 + self.b1
        h1 = self.relu(z1)

        logits = h1 @ self.W2 + self.b2
        probabilities = self.softmax(logits)

        cache = {
            "X": X,
            "z1": z1,
            "h1": h1,
            "probabilities": probabilities,
        }

        return probabilities, cache

    def backward(
        self,
        cache: dict[str, np.ndarray],
        labels: np.ndarray,
    ) -> dict[str, np.ndarray]:
        X = cache["X"]
        z1 = cache["z1"]
        h1 = cache["h1"]
        probabilities = cache["probabilities"]

        batch_size = X.shape[0]

        dlogits = probabilities.copy()

        dlogits[
            np.arange(batch_size),
            labels,
        ] -= 1.0

        dlogits /= batch_size

        dW2 = h1.T @ dlogits
        db2 = np.sum(
            dlogits,
            axis=0,
            keepdims=True,
        )

        dh1 = dlogits @ self.W2.T

        dz1 = dh1 * (z1 > 0)

        dW1 = X.T @ dz1
        db1 = np.sum(
            dz1,
            axis=0,
            keepdims=True,
        )

        return {
            "W1": dW1,
            "b1": db1,
            "W2": dW2,
            "b2": db2,
        }

    def update(
        self,
        gradients: dict[str, np.ndarray],
        *,
        learning_rate: float,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")

        self.W1 -= learning_rate * gradients["W1"]
        self.b1 -= learning_rate * gradients["b1"]
        self.W2 -= learning_rate * gradients["W2"]
        self.b2 -= learning_rate * gradients["b2"]

    def train_batch(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        *,
        learning_rate: float,
    ) -> float:
        probabilities, cache = self.forward(X)

        loss = self.cross_entropy_loss(
            probabilities,
            labels,
        )

        gradients = self.backward(
            cache,
            labels,
        )

        self.update(
            gradients,
            learning_rate=learning_rate,
        )

        return loss

    def predict_proba(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        probabilities, _ = self.forward(X)

        return probabilities

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        probabilities = self.predict_proba(X)

        return np.argmax(
            probabilities,
            axis=1,
        )

    def save(
        self,
        path: str,
    ) -> None:
        np.savez_compressed(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            input_dim=np.array(self.input_dim),
            hidden_dim=np.array(self.hidden_dim),
            num_classes=np.array(self.num_classes),
        )

    @classmethod
    def load(
        cls,
        path: str,
    ) -> "NumPyMLP":
        artifact = np.load(path)

        model = cls(
            input_dim=int(artifact["input_dim"]),
            hidden_dim=int(artifact["hidden_dim"]),
            num_classes=int(artifact["num_classes"]),
        )

        model.W1 = artifact["W1"].astype(np.float32)
        model.b1 = artifact["b1"].astype(np.float32)
        model.W2 = artifact["W2"].astype(np.float32)
        model.b2 = artifact["b2"].astype(np.float32)

        return model
