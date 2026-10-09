import torch
from torch.utils.data import DataLoader

from ml.src.models.textcnn import TextCNN
from ml.src.training.textcnn_trainer import (
    evaluate,
    train_epoch,
)


class TinyDataset:
    def __len__(self):
        return 6

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor(
                [2, 3, 4, 0, 0, 0],
                dtype=torch.long,
            ),
            "label": torch.tensor(
                index % 3,
                dtype=torch.long,
            ),
            "example_id": f"id_{index}",
            "text": f"text {index}",
            "label_name": [
                "sadness",
                "joy",
                "love",
            ][index % 3],
        }


def test_textcnn_train_epoch_returns_finite_loss():
    loader = DataLoader(
        TinyDataset(),
        batch_size=2,
        shuffle=False,
    )

    model = TextCNN(
        vocab_size=10,
        embedding_dim=8,
        num_filters=4,
        kernel_sizes=[2, 3],
        num_classes=3,
        dropout=0.0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
    )

    criterion = torch.nn.CrossEntropyLoss()

    loss = train_epoch(
        model,
        loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert isinstance(loss, float)
    assert loss > 0


def test_textcnn_evaluate_shapes():
    loader = DataLoader(
        TinyDataset(),
        batch_size=2,
        shuffle=False,
    )

    model = TextCNN(
        vocab_size=10,
        embedding_dim=8,
        num_filters=4,
        kernel_sizes=[2, 3],
        num_classes=3,
        dropout=0.0,
    )

    result = evaluate(
        model,
        loader,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    )

    assert result["predictions"].shape == (6,)
    assert result["probabilities"].shape == (
        6,
        3,
    )

    assert len(result["example_ids"]) == 6

    assert (result["probabilities"].sum(axis=1) > 0.999).all()

    assert (result["probabilities"].sum(axis=1) < 1.001).all()
