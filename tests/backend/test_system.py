from fastapi.testclient import TestClient

from backend.app.main import app


def test_models_returns_three_models():
    with TestClient(app) as client:
        response = client.get("/models")

    assert response.status_code == 200

    body = response.json()

    assert len(body["models"]) == 3

    ids = {model["model_id"] for model in body["models"]}

    assert ids == {
        "baseline_tfidf_dair_v001",
        "numpy_mlp_dair_v002",
        "textcnn_dair_v001",
    }


def test_only_textcnn_is_runtime_loaded():
    with TestClient(app) as client:
        response = client.get("/models")

    models = {item["model_id"]: item for item in response.json()["models"]}

    assert models["textcnn_dair_v001"]["runtime_loaded"] is True

    assert models["baseline_tfidf_dair_v001"]["runtime_loaded"] is False

    assert models["numpy_mlp_dair_v002"]["runtime_loaded"] is False


def test_textcnn_capabilities():
    with TestClient(app) as client:
        response = client.get("/models")

    models = {item["model_id"]: item for item in response.json()["models"]}

    capabilities = models["textcnn_dair_v001"]["capabilities"]

    assert "live_predict" in capabilities
    assert "live_interpret" in capabilities
    assert "error_lab" in capabilities
