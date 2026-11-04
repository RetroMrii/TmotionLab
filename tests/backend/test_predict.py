from fastapi.testclient import TestClient
from backend.app.main import app


def test_legacy_predict_is_explicitly_unavailable():
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "I feel happy.", "include_distribution": True})
    assert response.status_code == 503
    assert "Archived DAIR" in response.json()["error"]["message"]
    assert "top_prediction" not in response.json()


def test_predict_rejects_blank_text():
    with TestClient(app) as client:
        assert client.post("/predict", json={"text": "   "}).status_code == 422


def test_predict_rejects_missing_text():
    with TestClient(app) as client:
        assert client.post("/predict", json={}).status_code == 422
