from __future__ import annotations

import re

from ml.src.preprocessing.text import normalize_text


_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]",
    flags=re.UNICODE,
)


def tokenize(
    text: str,
    *,
    lowercase: bool = True,
) -> list[str]:
    normalized = normalize_text(
        text,
        lowercase=lowercase,
    )

    if not normalized:
        return []

    return _TOKEN_RE.findall(normalized)
