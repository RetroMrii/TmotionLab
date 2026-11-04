import torch
from torch.utils.data import DataLoader

from ml.src.data.goemotions_scheduled_dataset import (
    GoEmotionsScheduledDataset,
)
from ml.src.data.goemotions_dataset import (
    GoEmotionsCollator,
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

        return {
            "input_ids": torch.tensor(
                [
                    item["input_ids"] + [1] * (length - len(item["input_ids"]))
                    for item in features
                ]
            ),
            "attention_mask": torch.tensor(
                [
                    item["attention_mask"]
                    + [0] * (length - len(item["attention_mask"]))
                    for item in features
                ]
            ),
        }


def test_epoch_transition():
    rows, names = load_goemotions_rows("train")
    tokenizer = FakeTokenizer()

    dataset = GoEmotionsScheduledDataset(
        real_rows=rows,
        synthetic_rows=[],
        label_names=names,
        tokenizer=tokenizer,
        strategy="uniform",
    )

    dataset.set_epoch(0)
    first = dataset.schedule()

    dataset.set_epoch(1)
    second = dataset.schedule()

    dataset.set_epoch(0)
    repeated = dataset.schedule()

    assert len(dataset) == 43252
    assert len(first) == len(second) == 43252
    assert first != second
    assert first == repeated


def test_dataloader_shapes():
    rows, names = load_goemotions_rows("train")
    tokenizer = FakeTokenizer()

    dataset = GoEmotionsScheduledDataset(
        real_rows=rows,
        synthetic_rows=[],
        label_names=names,
        tokenizer=tokenizer,
        strategy="rare_oversample",
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
        collate_fn=GoEmotionsCollator(tokenizer),
    )

    batch = next(iter(loader))

    assert batch["input_ids"].shape[0] == 8
    assert batch["labels"].shape == (8, 28)
    assert batch["labels"].dtype == torch.float32
