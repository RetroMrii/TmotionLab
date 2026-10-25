import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data/processed/goemotions_v001"

EXPECTED_SPLITS = {
    "train": 43252,
    "validation": 5426,
    "test": 5427,
}

NUM_LABELS = 28


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def encode_multihot(label_ids, num_labels=NUM_LABELS):
    if not isinstance(label_ids, list) or not label_ids:
        raise ValueError("Expected non-empty label list")

    if len(set(label_ids)) != len(label_ids) or any(
        type(label) is not int or not 0 <= label < num_labels for label in label_ids
    ):
        raise ValueError("Invalid or duplicate label IDs")

    target = torch.zeros(num_labels, dtype=torch.float32)
    target[label_ids] = 1.0

    return target


def load_goemotions_rows(split: str):
    if split not in EXPECTED_SPLITS:
        raise ValueError(f"Unsupported split: {split}")

    manifest_path = DATA_DIR / "manifest.json"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if (
        manifest["dataset_version"] != "goemotions_v001"
        or len(manifest["label_names"]) != NUM_LABELS
    ):
        raise ValueError("Unexpected dataset manifest")

    info = manifest["processed_splits"][split]
    path = DATA_DIR / f"{split}.jsonl"

    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"{split}: checksum mismatch")

    rows = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue

            row = json.loads(line)

            if not isinstance(row["example_id"], str) or not row[
                "example_id"
            ].startswith(f"goe_{split}_"):
                raise ValueError("Invalid example ID")

            if row["example_id"] in seen_ids:
                raise ValueError("Duplicate example ID")

            seen_ids.add(row["example_id"])

            text = row["text"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Invalid text")

            actual_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

            if actual_hash != row["text_sha256"]:
                raise ValueError("Text hash mismatch")

            encode_multihot(row["labels"])
            rows.append(row)

    if len(rows) != info["rows"]:
        raise ValueError("Manifest row count mismatch")

    if len(rows) != EXPECTED_SPLITS[split]:
        raise ValueError("Unexpected canonical split size")

    return rows, manifest["label_names"]


class GoEmotionsDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        max_length=48,
    ):
        if not 2 <= max_length <= 512:
            raise ValueError("Invalid sequence length")

        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.example_ids = [row["example_id"] for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]

        encoded = self.tokenizer(
            row["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=True,
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": encode_multihot(row["labels"]),
        }


class GoEmotionsCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        token_features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            for item in features
        ]

        batch = self.tokenizer.pad(
            token_features,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.stack([item["labels"] for item in features]).to(
            torch.float32
        )

        return batch
