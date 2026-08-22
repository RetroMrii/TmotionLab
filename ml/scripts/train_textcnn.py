from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ml.src.data.text_dataset import EmotionTextDataset
from ml.src.models.textcnn import TextCNN
from ml.src.preprocessing.vocab import Vocabulary
from ml.src.training.textcnn_trainer import (
    evaluate,
    train_epoch,
)


LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TmotionLab TextCNN.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")

    return data


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def metrics_with_per_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
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
            for label in LABELS
        },
    }


def save_predictions(
    result: dict,
    *,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    probabilities = result["probabilities"]
    predictions = result["predictions"]

    frame = pd.DataFrame(
        {
            "example_id": result["example_ids"],
            "text": result["texts"],
            "true_label": result["labels"],
            "true_label_name": result["label_names"],
            "predicted_label": predictions,
            "predicted_label_name": [LABELS[int(label_id)] for label_id in predictions],
            "top_score": probabilities.max(axis=1),
        }
    )

    for label_id, label_name in enumerate(LABELS):
        frame[f"score_{label_name}"] = probabilities[:, label_id]

    frame.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    seed = int(config["seed"])
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device: {device}")

    preprocessing = config["preprocessing"]

    vocab = Vocabulary.load(preprocessing["vocab_path"])

    print(f"[INFO] Vocabulary size: {len(vocab)}")

    dataset_config = config["dataset"]
    processed_dir = Path(dataset_config["processed_dir"])

    dataset_kwargs = {
        "vocab": vocab,
        "max_sequence_length": int(preprocessing["max_sequence_length"]),
        "lowercase": bool(preprocessing["lowercase"]),
    }

    train_dataset = EmotionTextDataset(
        processed_dir / "train.csv",
        **dataset_kwargs,
    )

    validation_dataset = EmotionTextDataset(
        processed_dir / "validation.csv",
        **dataset_kwargs,
    )

    test_dataset = EmotionTextDataset(
        processed_dir / "test.csv",
        **dataset_kwargs,
    )

    training = config["training"]
    batch_size = int(training["batch_size"])

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model_config = config["model"]

    model = TextCNN(
        vocab_size=len(vocab),
        embedding_dim=int(model_config["embedding_dim"]),
        num_filters=int(model_config["num_filters"]),
        kernel_sizes=[int(value) for value in model_config["kernel_sizes"]],
        num_classes=int(model_config["num_classes"]),
        dropout=float(model_config["dropout"]),
        padding_idx=0,
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )

    artifacts = config["artifacts"]

    model_path = Path(artifacts["model"])
    metrics_path = Path(artifacts["metrics"])
    epoch_log_path = Path(artifacts["epoch_log"])
    validation_predictions_path = Path(artifacts["validation_predictions"])
    test_predictions_path = Path(artifacts["test_predictions"])
    config_snapshot_path = Path(artifacts["config_snapshot"])

    confusion_path = (
        metrics_path.parent.parent
        / "figures"
        / (config["experiment_id"] + "_confusion_matrix.png")
    )

    for path in [
        model_path,
        metrics_path,
        epoch_log_path,
        validation_predictions_path,
        test_predictions_path,
        config_snapshot_path,
        confusion_path,
    ]:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    patience = int(training["early_stopping_patience"])

    epoch_rows = []

    for epoch in range(
        1,
        int(training["epochs"]) + 1,
    ):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        validation_result = evaluate(
            model,
            validation_loader,
            criterion=criterion,
            device=device,
        )

        val_macro_f1 = validation_result["macro_f1"]

        epoch_rows.append(
            {
                "experiment_id": config["experiment_id"],
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": validation_result["loss"],
                "val_accuracy": validation_result["accuracy"],
                "val_macro_f1": val_macro_f1,
                "val_weighted_f1": (validation_result["weighted_f1"]),
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"[EPOCH {epoch:02d}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss="
            f"{validation_result['loss']:.4f} "
            f"val_acc="
            f"{validation_result['accuracy']:.4f} "
            f"val_macro_f1="
            f"{val_macro_f1:.4f}"
        )

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": (model.state_dict()),
                    "vocab_size": len(vocab),
                    "embedding_dim": int(model_config["embedding_dim"]),
                    "num_filters": int(model_config["num_filters"]),
                    "kernel_sizes": [
                        int(value) for value in model_config["kernel_sizes"]
                    ],
                    "num_classes": int(model_config["num_classes"]),
                    "dropout": float(model_config["dropout"]),
                    "max_sequence_length": int(preprocessing["max_sequence_length"]),
                    "best_epoch": best_epoch,
                },
                model_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"[INFO] Early stopping after epoch {epoch}")
            break

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

    checkpoint = torch.load(
        model_path,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    validation_result = evaluate(
        model,
        validation_loader,
        criterion=criterion,
        device=device,
    )

    test_result = evaluate(
        model,
        test_loader,
        criterion=criterion,
        device=device,
    )

    validation_metrics = metrics_with_per_class(
        validation_result["labels"],
        validation_result["predictions"],
    )

    test_metrics = metrics_with_per_class(
        test_result["labels"],
        test_result["predictions"],
    )

    save_predictions(
        validation_result,
        output_path=validation_predictions_path,
    )

    save_predictions(
        test_result,
        output_path=test_predictions_path,
    )

    metrics_payload = {
        "experiment_id": config["experiment_id"],
        "model_family": "textcnn",
        "seed": seed,
        "device": str(device),
        "best_epoch": best_epoch,
        "validation": validation_metrics,
        "test": test_metrics,
        "artifacts": {
            "model": str(model_path).replace("\\", "/"),
            "epoch_log": str(epoch_log_path).replace("\\", "/"),
            "validation_predictions": str(validation_predictions_path).replace(
                "\\", "/"
            ),
            "test_predictions": str(test_predictions_path).replace("\\", "/"),
            "confusion_matrix": str(confusion_path).replace("\\", "/"),
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
        test_result["labels"],
        test_result["predictions"],
        labels=list(range(len(LABELS))),
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=LABELS,
    )

    fig, ax = plt.subplots(figsize=(8, 7))

    display.plot(
        ax=ax,
        values_format="d",
    )

    ax.set_title("TextCNN — Test Confusion Matrix")

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

    print(f"[RESULT] Best epoch: {best_epoch}")
    print(f"[RESULT] Validation accuracy: {validation_metrics['accuracy']:.4f}")
    print(f"[RESULT] Validation macro F1: {validation_metrics['macro_f1']:.4f}")
    print(f"[RESULT] Test accuracy: {test_metrics['accuracy']:.4f}")
    print(f"[RESULT] Test macro F1: {test_metrics['macro_f1']:.4f}")
    print(f"[RESULT] Test weighted F1: {test_metrics['weighted_f1']:.4f}")

    print(f"[OK] Model saved: {model_path}")
    print(f"[OK] Metrics saved: {metrics_path}")
    print(f"[OK] Epoch log saved: {epoch_log_path}")
    print("[OK] TextCNN training complete")


if __name__ == "__main__":
    main()
