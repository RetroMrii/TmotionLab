import tempfile

import torch
from torch import nn
from torch.utils.data import IterableDataset
from transformers import Trainer, TrainingArguments


class EpochProbeDataset(IterableDataset):
    def __init__(self):
        super().__init__()
        self.epoch = 0
        self.set_calls = []
        self.iterations = []

    def __len__(self):
        return 8

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self.set_calls.append(self.epoch)
        print(f"[PROBE] set_epoch({self.epoch})")

    def __iter__(self):
        epoch = self.epoch
        self.iterations.append(epoch)
        print(f"[PROBE] iterating epoch {epoch}")

        for index in range(8):
            yield {
                "input_ids": torch.tensor([float(index), float(epoch)]),
                "labels": torch.tensor([float(index % 2)]),
            }


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, input_ids, labels=None):
        logits = self.linear(input_ids.float())

        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels.float(),
        )

        return {
            "loss": loss,
            "logits": logits,
        }


def main():
    dataset = EpochProbeDataset()

    with tempfile.TemporaryDirectory() as output_dir:
        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=2,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=1,
            learning_rate=0.001,
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            report_to="none",
            dataloader_num_workers=0,
            remove_unused_columns=False,
            use_cpu=True,
        )

        trainer = Trainer(
            model=TinyClassifier(),
            args=args,
            train_dataset=dataset,
        )

        trainer.train()

        print("set_epoch calls:", dataset.set_calls)
        print("iterated epochs:", dataset.iterations)
        print("Optimizer steps:", trainer.state.global_step)

        if 0 not in dataset.iterations:
            raise RuntimeError("Epoch 0 was not consumed")

        if 1 not in dataset.iterations:
            raise RuntimeError("Trainer did not advance to epoch 1")

        if trainer.state.global_step != 4:
            raise RuntimeError("Unexpected optimizer-step count")

        print("[OK] Trainer advances iterable dataset epochs")


if __name__ == "__main__":
    main()
