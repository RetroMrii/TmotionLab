from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from ml.src.models.textcnn import TextCNN


@dataclass
class EpochResult:
    train_loss: float
    val_accuracy: float
    val_macro_f1: float
    val_weighted_f1: float


def train_epoch(
    model: TextCNN,
    dataloader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


@torch.no_grad()
def evaluate(
    model: TextCNN,
    dataloader: DataLoader,
    *,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()

    total_loss = 0.0
    total_examples = 0

    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_example_ids = []
    all_texts = []
    all_label_names = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids)
        loss = criterion(logits, labels)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )

        predictions = torch.argmax(
            probabilities,
            dim=1,
        )

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_examples += batch_size

        all_labels.append(labels.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

        all_example_ids.extend(batch["example_id"])
        all_texts.extend(batch["text"])
        all_label_names.extend(batch["label_name"])

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_predictions)
    probabilities = np.concatenate(all_probabilities)

    return {
        "loss": total_loss / total_examples,
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
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
        "labels": y_true,
        "predictions": y_pred,
        "probabilities": probabilities,
        "example_ids": all_example_ids,
        "texts": all_texts,
        "label_names": all_label_names,
    }
