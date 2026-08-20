from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(
    text: str,
    *,
    lowercase: bool = True,
) -> str:
    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    normalized = _WHITESPACE_RE.sub(" ", text.strip())

    if lowercase:
        normalized = normalized.lower()

    return normalized
