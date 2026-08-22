import torch

from ml.src.interpretation.textcnn_interpreter import (
    interpret_textcnn_input,
)
from ml.src.models.textcnn import TextCNN


def test_interpretation_returns_valid_spans():
    model = TextCNN(
        vocab_size=20,
        embedding_dim=8,
        num_filters=4,
        kernel_sizes=[2, 3],
        num_classes=3,
        dropout=0.0,
    )

    input_ids = torch.tensor(
        [
            [
                2,
                3,
                4,
                5,
                0,
                0,
            ]
        ],
        dtype=torch.long,
    )

    tokens = [
        "i",
        "feel",
        "very",
        "happy",
    ]

    result = interpret_textcnn_input(
        model,
        input_ids=input_ids,
        tokens=tokens,
        kernel_sizes=[2, 3],
        labels=[
            "sadness",
            "joy",
            "love",
        ],
        top_k=5,
    )

    assert result["token_count"] == 4

    assert len(result["top_text_spans"]) <= 5

    for span in result["top_text_spans"]:
        assert span["token_end_exclusive"] <= len(tokens)

        assert span["contains_padding"] is False

        assert span["span_tokens"]


def test_interpretation_prediction_schema():
    model = TextCNN(
        vocab_size=20,
        embedding_dim=8,
        num_filters=4,
        kernel_sizes=[2, 3],
        num_classes=3,
        dropout=0.0,
    )

    input_ids = torch.tensor(
        [
            [
                2,
                3,
                4,
                5,
                0,
                0,
            ]
        ],
        dtype=torch.long,
    )

    result = interpret_textcnn_input(
        model,
        input_ids=input_ids,
        tokens=[
            "i",
            "feel",
            "very",
            "happy",
        ],
        kernel_sizes=[2, 3],
        labels=[
            "sadness",
            "joy",
            "love",
        ],
        top_k=3,
    )

    assert result["prediction"]["label"] in {
        "sadness",
        "joy",
        "love",
    }

    assert 0.0 <= result["prediction"]["score"] <= 1.0

    assert len(result["scores"]) == 3


def test_interpretation_tracks_padding_windows():
    model = TextCNN(
        vocab_size=20,
        embedding_dim=8,
        num_filters=4,
        kernel_sizes=[2, 3],
        num_classes=3,
        dropout=0.0,
    )

    input_ids = torch.tensor(
        [
            [
                2,
                3,
                0,
                0,
                0,
                0,
            ]
        ],
        dtype=torch.long,
    )

    result = interpret_textcnn_input(
        model,
        input_ids=input_ids,
        tokens=[
            "very",
            "happy",
        ],
        kernel_sizes=[2, 3],
        labels=[
            "sadness",
            "joy",
            "love",
        ],
        top_k=10,
    )

    assert result["padding_activation_count"] >= 0

    for span in result["top_text_spans"]:
        assert not span["contains_padding"]


def test_checkpoint_round_trip(tmp_path):
    from ml.src.interpretation.textcnn_interpreter import (
        load_textcnn_checkpoint,
    )

    model = TextCNN(
        vocab_size=20,
        embedding_dim=8,
        num_filters=4,
        kernel_sizes=[2, 3],
        num_classes=3,
        dropout=0.2,
    )

    path = tmp_path / "checkpoint.pt"

    torch.save(
        {
            "model_state_dict": (model.state_dict()),
            "vocab_size": 20,
            "embedding_dim": 8,
            "num_filters": 4,
            "kernel_sizes": [2, 3],
            "num_classes": 3,
            "dropout": 0.2,
            "max_sequence_length": 6,
        },
        path,
    )

    loaded, checkpoint = load_textcnn_checkpoint(
        path,
        device=torch.device("cpu"),
    )

    assert isinstance(
        loaded,
        TextCNN,
    )

    assert checkpoint["max_sequence_length"] == 6
