from fastapi.testclient import TestClient

from backend.app.main import app
from ml.src.inference.predictor import (
    LABELS,
)


def test_predict_returns_top_three():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": ("I feel extremely nervous about tomorrow.")},
        )

    assert response.status_code == 200

    body = response.json()

    assert body["model_id"] == "textcnn_dair_v001"

    assert len(body["top_3"]) == 3

    assert body["top_prediction"]["label"] in LABELS

    for item in body["top_3"]:
        assert item["label"] in LABELS

        assert 0.0 <= item["probability"] <= 1.0


def test_predict_distribution():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "text": "I feel happy.",
                "include_distribution": True,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["distribution"] is not None

    assert len(body["distribution"]) == 6

    total = sum(item["probability"] for item in body["distribution"])

    assert abs(total - 1.0) < 1e-5


def test_predict_rejects_blank_text():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "   "},
        )

    assert response.status_code == 422


def test_predict_rejects_missing_text():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={},
        )

    assert response.status_code == 422
