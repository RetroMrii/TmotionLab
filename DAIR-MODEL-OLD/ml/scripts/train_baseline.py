from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ml.src.models.baseline_tfidf import build_baseline_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the TmotionLab TF-IDF + Logistic Regression baseline."
    )
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
        raise ValueError(f"Expected YAML mapping in: {path}")

    return data


def load_split(
    processed_dir: Path,
    split_name: str,
) -> pd.DataFrame:
    path = processed_dir / f"{split_name}.csv"

    if not path.exists():
        raise FileNotFoundError(f"Processed split not found: {path}")

    return pd.read_csv(path)


def calculate_metrics(
    y_true,
    y_pred,
    *,
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

    per_class = {}

    for label_name in labels:
        class_report = report[label_name]

        per_class[label_name] = {
            "precision": float(class_report["precision"]),
            "recall": float(class_report["recall"]),
            "f1": float(class_report["f1-score"]),
            "support": int(class_report["support"]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
            )
        ),
        "per_class": per_class,
    }


def save_predictions(
    frame: pd.DataFrame,
    *,
    predictions,
    probabilities,
    model_classes,
    labels: list[str],
    output_path: Path,
    example_id_column: str,
    label_column: str,
    label_name_column: str,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = pd.DataFrame(
        {
            "example_id": frame[example_id_column],
            "text": frame["text"],
            "true_label": frame[label_column].astype(int),
            "true_label_name": frame[label_name_column],
            "predicted_label": predictions.astype(int),
            "predicted_label_name": [labels[int(label_id)] for label_id in predictions],
            "top_score": probabilities.max(axis=1),
        }
    )

    class_to_probability_column = {
        int(class_id): index for index, class_id in enumerate(model_classes)
    }

    for label_id, label_name in enumerate(labels):
        probability_index = class_to_probability_column[label_id]

        output[f"score_{label_name}"] = probabilities[:, probability_index]

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)

    dataset_config_path = Path(config["dataset"]["config"])

    dataset_config = load_yaml(dataset_config_path)

    labels = list(dataset_config["labels"])

    processed_dir = Path(config["dataset"]["processed_dir"])

    text_column = config["dataset"]["text_column"]
    label_column = config["dataset"]["label_column"]
    label_name_column = config["dataset"]["label_name_column"]
    example_id_column = config["dataset"]["example_id_column"]

    train = load_split(processed_dir, "train")
    validation = load_split(
        processed_dir,
        "validation",
    )
    test = load_split(processed_dir, "test")

    print(f"[INFO] Train rows: {len(train)}")
    print(f"[INFO] Validation rows: {len(validation)}")
    print(f"[INFO] Test rows: {len(test)}")

    pipeline = build_baseline_pipeline(
        config["vectorizer"],
        config["model"],
        seed=int(config["seed"]),
    )

    print("[INFO] Fitting TF-IDF + Logistic Regression...")

    pipeline.fit(
        train[text_column],
        train[label_column].astype(int),
    )

    print("[OK] Model fit complete")

    validation_predictions = pipeline.predict(validation[text_column])
    validation_probabilities = pipeline.predict_proba(validation[text_column])

    test_predictions = pipeline.predict(test[text_column])
    test_probabilities = pipeline.predict_proba(test[text_column])

    validation_metrics = calculate_metrics(
        validation[label_column].astype(int),
        validation_predictions,
        labels=labels,
    )

    test_metrics = calculate_metrics(
        test[label_column].astype(int),
        test_predictions,
        labels=labels,
    )

    artifacts = config["artifacts"]

    model_path = Path(artifacts["model"])
    metrics_path = Path(artifacts["metrics"])
    confusion_path = Path(artifacts["confusion_matrix"])
    validation_predictions_path = Path(artifacts["validation_predictions"])
    test_predictions_path = Path(artifacts["test_predictions"])
    config_snapshot_path = Path(artifacts["config_snapshot"])

    for path in [
        model_path,
        metrics_path,
        confusion_path,
        validation_predictions_path,
        test_predictions_path,
        config_snapshot_path,
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    joblib.dump(
        pipeline,
        model_path,
    )

    save_predictions(
        validation,
        predictions=validation_predictions,
        probabilities=validation_probabilities,
        model_classes=pipeline.named_steps["classifier"].classes_,
        labels=labels,
        output_path=validation_predictions_path,
        example_id_column=example_id_column,
        label_column=label_column,
        label_name_column=label_name_column,
    )

    save_predictions(
        test,
        predictions=test_predictions,
        probabilities=test_probabilities,
        model_classes=pipeline.named_steps["classifier"].classes_,
        labels=labels,
        output_path=test_predictions_path,
        example_id_column=example_id_column,
        label_column=label_column,
        label_name_column=label_name_column,
    )

    metrics_payload = {
        "experiment_id": config["experiment_id"],
        "dataset": dataset_config["dataset_name"],
        "task_type": dataset_config["task_type"],
        "model_family": "tfidf_logistic_regression",
        "seed": int(config["seed"]),
        "validation": validation_metrics,
        "test": test_metrics,
        "artifacts": {
            "model": str(model_path).replace("\\", "/"),
            "confusion_matrix": str(confusion_path).replace("\\", "/"),
            "validation_predictions": str(validation_predictions_path).replace(
                "\\", "/"
            ),
            "test_predictions": str(test_predictions_path).replace("\\", "/"),
            "config_snapshot": str(config_snapshot_path).replace("\\", "/"),
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

    matrix = confusion_matrix(
        test[label_column].astype(int),
        test_predictions,
        labels=list(range(len(labels))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=labels,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title("TF-IDF + Logistic Regression — Test Confusion Matrix")

    fig.tight_layout()

    fig.savefig(
        confusion_path,
        dpi=150,
    )

    plt.close(fig)

    shutil.copy2(
        args.config,
        config_snapshot_path,
    )

    print(f"[RESULT] Validation accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"[RESULT] Validation macro F1: {validation_metrics['macro_f1']:.4f}")

    print(f"[RESULT] Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"[RESULT] Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"[RESULT] Test weighted F1: {test_metrics['weighted_f1']:.4f}")

    print(f"[OK] Model saved: {model_path}")
    print(f"[OK] Metrics saved: {metrics_path}")
    print(f"[OK] Test predictions saved: {test_predictions_path}")
    print(f"[OK] Confusion matrix saved: {confusion_path}")
    print("[OK] Baseline training complete")


if __name__ == "__main__":
    main()
