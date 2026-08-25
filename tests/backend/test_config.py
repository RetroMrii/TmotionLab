from backend.app.config import (
    _parse_origins,
    load_settings,
)


def test_default_settings():
    settings = load_settings()

    assert settings.api_title == "TmotionLab API"

    assert settings.api_version == "0.3.0"


def test_parse_origins():
    result = _parse_origins("http://localhost:5173,http://localhost:3000")

    assert result == (
        "http://localhost:5173",
        "http://localhost:3000",
    )
