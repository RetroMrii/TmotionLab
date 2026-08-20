import pytest

from ml.src.preprocessing.text import normalize_text
from ml.src.preprocessing.tokenizer import tokenize


def test_normalize_collapses_whitespace():
    text = "  I   feel\nvery\tgood  "

    assert normalize_text(text) == "i feel very good"


def test_normalize_can_preserve_case():
    text = "I Feel GOOD"

    assert (
        normalize_text(
            text,
            lowercase=False,
        )
        == "I Feel GOOD"
    )


def test_tokenizer_preserves_contractions():
    assert tokenize("I'm nervous") == [
        "i'm",
        "nervous",
    ]


def test_tokenizer_preserves_punctuation():
    assert tokenize("Really?!") == [
        "really",
        "?",
        "!",
    ]


def test_tokenizer_empty_text():
    assert tokenize("   ") == []


def test_normalizer_rejects_non_string():
    with pytest.raises(TypeError):
        normalize_text(123)
