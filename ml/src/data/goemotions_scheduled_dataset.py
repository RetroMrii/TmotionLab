import hashlib

from torch.utils.data import IterableDataset

from ml.src.data.goemotions_dataset import encode_multihot
from ml.src.data.goemotions_experiment_sampling import (
    REAL_DRAWS,
    make_epoch_schedule,
)


class GoEmotionsScheduledDataset(IterableDataset):
    def __init__(
        self,
        real_rows,
        synthetic_rows,
        label_names,
        tokenizer,
        strategy,
        seed=42,
        max_length=48,
    ):
        super().__init__()

        if len(real_rows) != REAL_DRAWS:
            raise ValueError("Unexpected real dataset size")

        if strategy not in (
            "uniform",
            "rare_oversample",
            "synthetic",
            "hybrid",
        ):
            raise ValueError("Unsupported strategy")

        if not 2 <= max_length <= 512:
            raise ValueError("Invalid sequence limit")

        self.real_rows = real_rows
        self.synthetic_rows = synthetic_rows
        self.label_names = label_names
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.seed = seed
        self.max_length = max_length
        self.epoch = 0
        self.epoch_set_history = []
        self.epoch_iteration_history = []

    def __len__(self):
        return REAL_DRAWS

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Invalid epoch")

        self.epoch = epoch
        self.epoch_set_history.append(epoch)

        print(
            f"[SAMPLER] strategy={self.strategy} set_epoch={epoch}",
            flush=True,
        )

    def schedule(self):
        return make_epoch_schedule(
            self.real_rows,
            self.synthetic_rows,
            self.label_names,
            self.strategy,
            self.seed,
            self.epoch,
        )

    def __iter__(self):

        self.epoch_iteration_history.append(self.epoch)
        print(
            f"[SAMPLER] strategy={self.strategy} consuming_epoch={self.epoch}",
            flush=True,
        )

        for source, index in self.schedule():
            rows = self.real_rows if source == "real" else self.synthetic_rows

            row = rows[index]

            encoded = self.tokenizer(
                row["text"],
                add_special_tokens=True,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_attention_mask=True,
            )

            yield {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": encode_multihot(row["labels"]),
            }
