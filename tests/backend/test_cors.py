from fastapi.testclient import TestClient

from backend.app.main import app


def test_vite_origin_is_allowed():
    with TestClient(app) as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": ("http://localhost:5173"),
                ("Access-Control-Request-Method"): "POST",
            },
        )

    assert response.status_code == 200

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_unknown_origin_not_allowed():
    with TestClient(app) as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": ("https://not-allowed.example"),
                ("Access-Control-Request-Method"): "POST",
            },
        )

    assert "access-control-allow-origin" not in response.headers
