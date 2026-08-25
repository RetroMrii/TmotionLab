from fastapi.testclient import TestClient

from backend.app.main import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ok"

    assert body["service"] == "TmotionLab API"

    assert body["textcnn_loaded"] is True

    assert body["model_id"] == "textcnn_dair_v001"

    assert body["artifacts_loaded"] is True

    assert body["version"] == "0.3.0"
