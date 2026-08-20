import pytest

from ml.src.preprocessing.vocab import (
    PAD_ID,
    PAD_TOKEN,
    UNK_ID,
    UNK_TOKEN,
    Vocabulary,
)


def test_special_tokens_are_fixed():
    vocab = Vocabulary()

    assert vocab.lookup_token(PAD_TOKEN) == PAD_ID
    assert vocab.lookup_token(UNK_TOKEN) == UNK_ID

    assert vocab.lookup_id(PAD_ID) == PAD_TOKEN
    assert vocab.lookup_id(UNK_ID) == UNK_TOKEN


def test_vocab_builds_by_frequency_then_alphabetically():
    vocab = Vocabulary()

    vocab.build(
        [
            ["joy", "fear", "joy"],
            ["anger", "fear"],
            ["joy"],
        ]
    )

    assert vocab.id_to_token == [
        "<PAD>",
        "<UNK>",
        "joy",
        "fear",
        "anger",
    ]


def test_unknown_token_maps_to_unk():
    vocab = Vocabulary()
    vocab.build([["joy"]])

    assert vocab.lookup_token("missing") == UNK_ID


def test_encode_tokens():
    vocab = Vocabulary()
    vocab.build([["joy", "fear"]])

    encoded = vocab.encode(["joy", "unknown", "fear"])

    assert encoded == [
        vocab.lookup_token("joy"),
        UNK_ID,
        vocab.lookup_token("fear"),
    ]


def test_min_frequency():
    vocab = Vocabulary(min_freq=2)

    vocab.build(
        [
            ["joy", "joy"],
            ["fear"],
        ]
    )

    assert "joy" in vocab.token_to_id
    assert "fear" not in vocab.token_to_id


def test_max_size_includes_special_tokens():
    vocab = Vocabulary(max_size=4)

    vocab.build(
        [
            ["joy", "joy", "joy"],
            ["fear", "fear"],
            ["anger"],
        ]
    )

    assert len(vocab) == 4

    assert vocab.id_to_token == [
        "<PAD>",
        "<UNK>",
        "joy",
        "fear",
    ]


def test_invalid_max_size():
    with pytest.raises(ValueError):
        Vocabulary(max_size=1)


def test_vocab_save_and_load(tmp_path):
    vocab = Vocabulary(
        max_size=10,
        min_freq=1,
    )

    vocab.build(
        [
            ["joy", "fear", "joy"],
        ]
    )

    path = tmp_path / "vocab.json"

    vocab.save(path)

    loaded = Vocabulary.load(path)

    assert loaded.token_to_id == vocab.token_to_id
    assert loaded.id_to_token == vocab.id_to_token
    assert loaded.max_size == vocab.max_size
    assert loaded.min_freq == vocab.min_freq
