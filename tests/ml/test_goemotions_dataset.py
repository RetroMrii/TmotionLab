import pytest
import torch

from ml.src.data.goemotions_dataset import (
    GoEmotionsDataset,
    GoEmotionsCollator,
    encode_multihot,
    load_goemotions_rows,
)


class FakeTokenizer:
    def __call__(self, text, **kwargs):
        ids = [0] + [10] * len(text.split()) + [2]
        ids = ids[: kwargs["max_length"]]

        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
        }

    def pad(self, features, padding, return_tensors):
        length = max(len(x["input_ids"]) for x in features)

        ids = []
        masks = []

        for item in features:
            pad_count = length - len(item["input_ids"])

            ids.append(item["input_ids"] + [1] * pad_count)
            masks.append(item["attention_mask"] + [0] * pad_count)

        return {
            "input_ids": torch.tensor(ids),
            "attention_mask": torch.tensor(masks),
        }


def test_multihot_encoding():
    target = encode_multihot([2, 7, 27])

    assert target.shape == (28,)
    assert target.dtype == torch.float32
    assert target.sum().item() == 3
    assert target[2] == 1
    assert target[7] == 1
    assert target[27] == 1


@pytest.mark.parametrize(
    "labels",
    [[], [28], [-1], [1, 1], [1.5], ["joy"]],
)
def test_invalid_targets_rejected(labels):
    with pytest.raises(ValueError):
        encode_multihot(labels)


def test_dataset_item_and_dynamic_padding():
    tokenizer = FakeTokenizer()

    rows = [
        {
            "example_id": "goe_train_000000",
            "text": "hello world",
            "labels": [1, 2],
        },
        {
            "example_id": "goe_train_000001",
            "text": "short",
            "labels": [3],
        },
    ]

    dataset = GoEmotionsDataset(rows, tokenizer, max_length=48)

    assert len(dataset) == 2

    collator = GoEmotionsCollator(tokenizer)

    batch = collator(
        [
            dataset[0],
            dataset[1],
        ]
    )

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["labels"].shape == (2, 28)
    assert batch["labels"].dtype == torch.float32
    assert batch["attention_mask"][1, -1] == 0

    assert dataset.example_ids == [
        "goe_train_000000",
        "goe_train_000001",
    ]


def test_canonical_dataset_loading():
    train, train_labels = load_goemotions_rows("train")
    val, val_labels = load_goemotions_rows("validation")

    assert len(train) == 43252
    assert len(val) == 5426
    assert train_labels == val_labels
    assert len(train_labels) == 28

    assert train[0]["example_id"] == "goe_train_000000"
    assert val[0]["example_id"] == "goe_validation_000000"
