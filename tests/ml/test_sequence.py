import pytest

from ml.src.preprocessing.sequence import (
    encode_text,
    pad_or_truncate,
)
from ml.src.preprocessing.vocab import (
    PAD_ID,
    UNK_ID,
    Vocabulary,
)


def test_padding():
    result = pad_or_truncate(
        [2, 3],
        max_length=5,
    )

    assert result == [
        2,
        3,
        PAD_ID,
        PAD_ID,
        PAD_ID,
    ]


def test_truncation():
    result = pad_or_truncate(
        [2, 3, 4, 5],
        max_length=2,
    )

    assert result == [2, 3]


def test_exact_length():
    result = pad_or_truncate(
        [2, 3, 4],
        max_length=3,
    )

    assert result == [2, 3, 4]


def test_invalid_max_length():
    with pytest.raises(ValueError):
        pad_or_truncate(
            [2, 3],
            max_length=0,
        )


def test_encode_text():
    vocab = Vocabulary()

    vocab.build(
        [
            ["joy", "fear"],
        ]
    )

    result = encode_text(
        ["joy", "missing"],
        vocab,
        max_length=4,
    )

    assert result == [
        vocab.lookup_token("joy"),
        UNK_ID,
        PAD_ID,
        PAD_ID,
    ]


def test_expected_model_sequence_length():
    max_sequence_length = 64

    assert (
        len(
            pad_or_truncate(
                [2, 3, 4],
                max_length=max_sequence_length,
            )
        )
        == 64
    )
