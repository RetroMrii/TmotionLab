from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.goemotions import MODEL_ID


def test_registry_separates_tasks_and_capabilities():
    with TestClient(app) as client:
        response = client.get("/models")
    assert response.status_code == 200
    models = {m["model_id"]: m for m in response.json()["models"]}
    assert len(models) == 4
    production = models.pop(MODEL_ID)
    assert production["runtime_loaded"] is True
    assert production["task"] == "multi_label"
    assert production["capabilities"] == ["live_predict"]
    assert production["accuracy"] is None
    assert len(production["labels"]) == 28
    for model in models.values():
        assert model["runtime_loaded"] is False
        assert model["dataset"] == "dair_ai_emotion"
        assert "live_predict" not in model["capabilities"]
        assert "live_interpret" not in model["capabilities"]
        assert "metrics" in model["capabilities"]
    assert "error_lab" in models["textcnn_dair_v001"]["capabilities"]
