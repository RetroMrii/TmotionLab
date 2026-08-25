from fastapi.testclient import TestClient

from backend.app.main import app


def test_unknown_compare_error_shape():
    with TestClient(app) as client:
        response = client.get("/compare/not_real")

    assert response.status_code == 404

    body = response.json()

    assert body == {
        "error": {
            "code": "http_404",
            "message": ("Unknown example_id: not_real"),
            "details": None,
        }
    }


def test_validation_error_shape():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={},
        )

    assert response.status_code == 422

    body = response.json()

    assert body["error"]["code"] == "validation_error"

    assert body["error"]["message"] == "Request validation failed"

    assert isinstance(
        body["error"]["details"],
        list,
    )


def test_blank_predict_error_shape():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "   "},
        )

    assert response.status_code == 422

    body = response.json()

    assert body["error"]["code"] == "http_422"

    assert body["error"]["message"] == "text must not be empty"
