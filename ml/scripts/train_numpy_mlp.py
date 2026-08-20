from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

from ml.src.models.numpy_mlp import NumPyMLP
from ml.src.training.numpy_trainer import (
    evaluate_sparse,
    train_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TmotionLab NumPy MLP.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")

    return data


def load_split(
    processed_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    path = processed_dir / f"{split_name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing processed split: {path}")

    return pd.read_csv(path)


def save_predictions(
    frame: pd.DataFrame,
    *,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = pd.DataFrame(
        {
            "example_id": frame["example_id"],
            "text": frame["text"],
            "true_label": frame["label"].astype(int),
            "true_label_name": frame["label_name"],
            "predicted_label": predictions.astype(int),
            "predicted_label_name": [labels[int(label_id)] for label_id in predictions],
            "top_score": probabilities.max(axis=1),
        }
    )

    for label_id, label_name in enumerate(labels):
        output[f"score_{label_name}"] = probabilities[:, label_id]

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def metrics_with_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": float(report["accuracy"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in labels
        },
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    processed_dir = Path(config["dataset"]["processed_dir"])

    train = load_split(processed_dir, "train")
    validation = load_split(
        processed_dir,
        "validation",
    )
    test = load_split(processed_dir, "test")

    labels = [
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise",
    ]

    seed = int(config["seed"])
    rng = np.random.default_rng(seed)

    vectorizer_config = config["vectorizer"]

    vectorizer = TfidfVectorizer(
        lowercase=bool(vectorizer_config["lowercase"]),
        ngram_range=tuple(vectorizer_config["ngram_range"]),
        min_df=vectorizer_config["min_df"],
        max_df=vectorizer_config["max_df"],
        max_features=vectorizer_config["max_features"],
        sublinear_tf=bool(vectorizer_config["sublinear_tf"]),
        dtype=np.float32,
    )

    print("[INFO] Fitting TF-IDF on training split only")

    X_train = vectorizer.fit_transform(train["text"])

    X_validation = vectorizer.transform(validation["text"])

    X_test = vectorizer.transform(test["text"])

    y_train = train["label"].to_numpy(dtype=np.int64)

    y_validation = validation["label"].to_numpy(dtype=np.int64)

    y_test = test["label"].to_numpy(dtype=np.int64)

    print(f"[INFO] TF-IDF feature count: {X_train.shape[1]}")

    model = NumPyMLP(
        input_dim=X_train.shape[1],
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_classes=int(config["model"]["num_classes"]),
        seed=seed,
    )

    training_config = config["training"]

    epochs = int(training_config["epochs"])
    batch_size = int(training_config["batch_size"])
    learning_rate = float(training_config["learning_rate"])

    artifacts = config["artifacts"]

    model_path = Path(artifacts["model"])
    vectorizer_path = Path(artifacts["vectorizer"])
    metrics_path = Path(artifacts["metrics"])
    epoch_log_path = Path(artifacts["epoch_log"])
    validation_predictions_path = Path(artifacts["validation_predictions"])
    test_predictions_path = Path(artifacts["test_predictions"])
    config_snapshot_path = Path(artifacts["config_snapshot"])

    for path in [
        model_path,
        vectorizer_path,
        metrics_path,
        epoch_log_path,
        validation_predictions_path,
        test_predictions_path,
        config_snapshot_path,
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    best_macro_f1 = -1.0
    best_epoch = -1

    epoch_rows = []

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            X_train,
            y_train,
            batch_size=batch_size,
            learning_rate=learning_rate,
            rng=rng,
        )

        validation_result = evaluate_sparse(
            model,
            X_validation,
            y_validation,
        )

        val_accuracy = validation_result["accuracy"]
        val_macro_f1 = validation_result["macro_f1"]
        val_weighted_f1 = validation_result["weighted_f1"]

        epoch_rows.append(
            {
                "experiment_id": config["experiment_id"],
                "epoch": epoch,
                "train_loss": train_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": (val_weighted_f1),
                "learning_rate": learning_rate,
            }
        )

        print(
            f"[EPOCH {epoch:02d}] "
            f"loss={train_loss:.4f} "
            f"val_acc={val_accuracy:.4f} "
            f"val_macro_f1={val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch

            model.save(str(model_path))

    with epoch_log_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=epoch_rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(epoch_rows)

    joblib.dump(
        vectorizer,
        vectorizer_path,
    )

    best_model = NumPyMLP.load(str(model_path))

    validation_result = evaluate_sparse(
        best_model,
        X_validation,
        y_validation,
    )

    test_result = evaluate_sparse(
        best_model,
        X_test,
        y_test,
    )

    validation_metrics = metrics_with_per_class(
        y_validation,
        validation_result["predictions"],
        labels,
    )

    test_metrics = metrics_with_per_class(
        y_test,
        test_result["predictions"],
        labels,
    )

    save_predictions(
        validation,
        predictions=validation_result["predictions"],
        probabilities=validation_result["probabilities"],
        labels=labels,
        output_path=validation_predictions_path,
    )

    save_predictions(
        test,
        predictions=test_result["predictions"],
        probabilities=test_result["probabilities"],
        labels=labels,
        output_path=test_predictions_path,
    )

    metrics_payload = {
        "experiment_id": config["experiment_id"],
        "model_family": "numpy_mlp",
        "input_representation": "tfidf_dense_minibatches",
        "seed": seed,
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
        "artifacts": {
            "model": str(model_path).replace("\\", "/"),
            "vectorizer": str(vectorizer_path).replace("\\", "/"),
            "epoch_log": str(epoch_log_path).replace("\\", "/"),
            "validation_predictions": str(validation_predictions_path).replace(
                "\\", "/"
            ),
            "test_predictions": str(test_predictions_path).replace("\\", "/"),
        },
    }

    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metrics_payload,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    shutil.copy2(
        args.config,
        config_snapshot_path,
    )

    print(f"[RESULT] Best epoch: {best_epoch}")
    print(f"[RESULT] Validation accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"[RESULT] Validation macro F1: {validation_metrics['macro_f1']:.4f}")
    print(f"[RESULT] Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"[RESULT] Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"[RESULT] Test weighted F1: {test_metrics['weighted_f1']:.4f}")

    print(f"[OK] Model saved: {model_path}")
    print(f"[OK] Vectorizer saved: {vectorizer_path}")
    print(f"[OK] Epoch log saved: {epoch_log_path}")
    print("[OK] NumPy MLP training complete")


if __name__ == "__main__":
    main()
