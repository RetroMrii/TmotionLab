from __future__ import annotations

import os
from dataclasses import dataclass


def _parse_origins(
    value: str | None,
) -> tuple[str, ...]:
    if value is None:
        return (
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        )

    origins = tuple(origin.strip() for origin in value.split(",") if origin.strip())

    if not origins:
        raise ValueError("TMOTIONLAB_CORS_ORIGINS must contain at least one origin")

    return origins


@dataclass(frozen=True)
class Settings:
    api_title: str
    api_version: str
    api_description: str
    cors_origins: tuple[str, ...]


def load_settings() -> Settings:
    return Settings(
        api_title="TmotionLab API",
        api_version="0.3.0",
        api_description=(
            "Inference and analysis API for the TmotionLab NLP laboratory."
        ),
        cors_origins=_parse_origins(os.getenv("TMOTIONLAB_CORS_ORIGINS")),
    )


settings = load_settings()
