from __future__ import annotations
from collections import Counter
from collections.abc import Iterable
import json
from pathlib import Path

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
UNK_ID = 1


class Vocabulary:
    def __init__(
        self,
        *,
        max_size: int | None = None,
        min_freq: int = 1,
    ) -> None:
        if max_size is not None and max_size < 2:
            raise ValueError("max_size must be at least 2 to include <PAD> and <UNK>")

        if min_freq < 1:
            raise ValueError("min_freq must be >= 1")

        self.max_size = max_size
        self.min_freq = min_freq

        self.token_to_id: dict[str, int] = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
        }

        self.id_to_token: list[str] = [
            PAD_TOKEN,
            UNK_TOKEN,
        ]

    def __len__(self) -> int:
        return len(self.id_to_token)

    def build(
        self,
        token_sequences: Iterable[Iterable[str]],
    ) -> None:
        counter: Counter[str] = Counter()

        for tokens in token_sequences:
            counter.update(tokens)

        candidates = [
            (token, count)
            for token, count in counter.items()
            if count >= self.min_freq and token not in {PAD_TOKEN, UNK_TOKEN}
        ]

        candidates.sort(key=lambda item: (-item[1], item[0]))

        if self.max_size is not None:
            available = self.max_size - 2
            candidates = candidates[:available]

        self.token_to_id = {
            PAD_TOKEN: PAD_ID,
            UNK_TOKEN: UNK_ID,
        }

        self.id_to_token = [
            PAD_TOKEN,
            UNK_TOKEN,
        ]

        for token, _ in candidates:
            token_id = len(self.id_to_token)

            self.token_to_id[token] = token_id
            self.id_to_token.append(token)

    def lookup_token(self, token: str) -> int:
        return self.token_to_id.get(
            token,
            UNK_ID,
        )

    def lookup_id(self, token_id: int) -> str:
        if token_id < 0 or token_id >= len(self.id_to_token):
            raise IndexError(f"token_id out of range: {token_id}")

        return self.id_to_token[token_id]

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.lookup_token(token) for token in tokens]

    def to_dict(self) -> dict:
        return {
            "max_size": self.max_size,
            "min_freq": self.min_freq,
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.to_dict(),
                handle,
                indent=2,
                ensure_ascii=False,
            )

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        input_path = Path(path)

        with input_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        vocab = cls(
            max_size=data["max_size"],
            min_freq=data["min_freq"],
        )

        vocab.token_to_id = {
            str(token): int(token_id) for token, token_id in data["token_to_id"].items()
        }

        vocab.id_to_token = [str(token) for token in data["id_to_token"]]

        return vocab
