from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from ml.src.preprocessing.sequence import encode_text
from ml.src.preprocessing.tokenizer import tokenize
from ml.src.preprocessing.vocab import Vocabulary


class EmotionTextDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        *,
        vocab: Vocabulary,
        max_sequence_length: int,
        lowercase: bool = True,
    ) -> None:
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")

        if max_sequence_length < 1:
            raise ValueError("max_sequence_length must be >= 1")

        self.frame = pd.read_csv(self.csv_path)

        required_columns = {
            "example_id",
            "text",
            "label",
            "label_name",
        }

        missing_columns = required_columns - set(self.frame.columns)

        if missing_columns:
            raise ValueError(f"Missing dataset columns: {sorted(missing_columns)}")

        self.vocab = vocab
        self.max_sequence_length = max_sequence_length
        self.lowercase = lowercase

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]

        text = str(row["text"])

        tokens = tokenize(
            text,
            lowercase=self.lowercase,
        )

        input_ids = encode_text(
            tokens,
            self.vocab,
            max_length=self.max_sequence_length,
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "label": torch.tensor(
                int(row["label"]),
                dtype=torch.long,
            ),
            "example_id": str(row["example_id"]),
            "text": text,
            "label_name": str(row["label_name"]),
        }
