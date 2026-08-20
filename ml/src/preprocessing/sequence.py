from __future__ import annotations

from collections.abc import Iterable

from ml.src.preprocessing.vocab import PAD_ID, Vocabulary


def pad_or_truncate(
    token_ids: Iterable[int],
    *,
    max_length: int,
    pad_id: int = PAD_ID,
) -> list[int]:
    if max_length < 1:
        raise ValueError("max_length must be >= 1")

    ids = list(token_ids)

    if len(ids) >= max_length:
        return ids[:max_length]

    padding_needed = max_length - len(ids)

    return ids + [pad_id] * padding_needed


def encode_text(
    tokens: Iterable[str],
    vocab: Vocabulary,
    *,
    max_length: int,
) -> list[int]:
    token_ids = vocab.encode(tokens)

    return pad_or_truncate(
        token_ids,
        max_length=max_length,
    )
