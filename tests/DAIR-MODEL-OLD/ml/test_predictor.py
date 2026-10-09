import torch

from ml.src.inference.predictor import (
    LABELS,
    TextCNNPredictor,
)


def test_real_textcnn_predictor():
    predictor = TextCNNPredictor(
        "ml/configs/textcnn_dair_v001.yaml",
        device=torch.device("cpu"),
    )

    result = predictor.predict("I feel very happy today.")

    assert result["model_id"] == "textcnn_dair_v001"

    assert result["mode"] == "single_label"

    assert len(result["top_3"]) == 3

    assert result["top_prediction"]["label"] in LABELS

    probabilities = [item["probability"] for item in result["top_3"]]

    assert all(0.0 <= value <= 1.0 for value in probabilities)

    assert probabilities == sorted(
        probabilities,
        reverse=True,
    )


def test_predictor_full_distribution():
    predictor = TextCNNPredictor(
        "ml/configs/textcnn_dair_v001.yaml",
        device=torch.device("cpu"),
    )

    result = predictor.predict(
        "I am nervous.",
        include_distribution=True,
    )

    assert len(result["distribution"]) == len(LABELS)

    total = sum(item["probability"] for item in result["distribution"])

    assert abs(total - 1.0) < 1e-5


def test_predictor_rejects_empty_text():
    predictor = TextCNNPredictor(
        "ml/configs/textcnn_dair_v001.yaml",
        device=torch.device("cpu"),
    )

    try:
        predictor.predict("   ")
        assert False
    except ValueError:
        pass
