import pandas as pd
import torch

from ml.src.data.text_dataset import (
    EmotionTextDataset,
)
from ml.src.preprocessing.vocab import (
    PAD_ID,
    Vocabulary,
)


def test_text_dataset_output(tmp_path):
    frame = pd.DataFrame(
        {
            "example_id": [
                "example_001",
                "example_002",
            ],
            "text": [
                "I feel happy!",
                "I feel afraid.",
            ],
            "label": [1, 4],
            "label_name": [
                "joy",
                "fear",
            ],
        }
    )

    csv_path = tmp_path / "sample.csv"

    frame.to_csv(
        csv_path,
        index=False,
    )

    vocab = Vocabulary()

    vocab.build(
        [
            ["i", "feel", "happy", "!"],
            ["afraid", "."],
        ]
    )

    dataset = EmotionTextDataset(
        csv_path,
        vocab=vocab,
        max_sequence_length=8,
        lowercase=True,
    )

    item = dataset[0]

    assert len(dataset) == 2

    assert item["input_ids"].shape == (8,)
    assert item["input_ids"].dtype == torch.long

    assert item["label"].item() == 1
    assert item["example_id"] == "example_001"
    assert item["label_name"] == "joy"


def test_text_dataset_applies_padding(tmp_path):
    frame = pd.DataFrame(
        {
            "example_id": ["example_001"],
            "text": ["happy"],
            "label": [1],
            "label_name": ["joy"],
        }
    )

    csv_path = tmp_path / "sample.csv"

    frame.to_csv(
        csv_path,
        index=False,
    )

    vocab = Vocabulary()
    vocab.build([["happy"]])

    dataset = EmotionTextDataset(
        csv_path,
        vocab=vocab,
        max_sequence_length=5,
    )

    ids = dataset[0]["input_ids"]

    assert ids.shape == (5,)
    assert ids[-1].item() == PAD_ID
