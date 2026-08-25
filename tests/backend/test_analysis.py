from fastapi.testclient import TestClient

from backend.app.main import app


def test_metrics_returns_three_models():
    with TestClient(app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200

    body = response.json()

    assert len(body["models"]) == 3

    model_ids = {item["model_id"] for item in body["models"]}

    assert model_ids == {
        "baseline_tfidf_dair_v001",
        "numpy_mlp_dair_v002",
        "textcnn_dair_v001",
    }


def test_metrics_reproduce_textcnn():
    with TestClient(app) as client:
        response = client.get("/metrics")

    models = {item["model_id"]: item for item in response.json()["models"]}

    textcnn = models["textcnn_dair_v001"]

    assert abs(textcnn["accuracy"] - 0.8965) < 1e-4

    assert abs(textcnn["macro_f1"] - 0.8510) < 1e-4


def test_compare_known_example():
    with TestClient(app) as client:
        response = client.get("/compare/dair_test_001837")

    assert response.status_code == 200

    body = response.json()

    assert body["example_id"] == "dair_test_001837"

    assert body["true_label_name"] == "sadness"

    assert body["textcnn_prediction"] == "anger"


def test_compare_unknown_example():
    with TestClient(app) as client:
        response = client.get("/compare/not_a_real_example")

    assert response.status_code == 404


def test_interpret():
    with TestClient(app) as client:
        response = client.post(
            "/interpret",
            json={
                "text": ("I feel extremely nervous about tomorrow."),
                "top_k": 5,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["model_id"] == "textcnn_dair_v001"

    assert len(body["top_text_spans"]) <= 5

    assert body["prediction"]["label"] in {
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise",
    }


def test_interpret_rejects_blank_text():
    with TestClient(app) as client:
        response = client.post(
            "/interpret",
            json={"text": "   "},
        )

    assert response.status_code == 422


def test_errors_default_page():
    with TestClient(app) as client:
        response = client.get("/errors")

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 207
    assert body["limit"] == 50
    assert len(body["items"]) == 50


def test_errors_shared_only():
    with TestClient(app) as client:
        response = client.get(
            "/errors",
            params={
                "shared_only": True,
                "limit": 200,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 93

    assert all(item["all_agree_and_wrong"] for item in body["items"])


def test_errors_baseline_regressions():
    with TestClient(app) as client:
        response = client.get(
            "/errors",
            params={
                "baseline_regression_only": True,
                "limit": 200,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 76


def test_errors_score_filter():
    with TestClient(app) as client:
        response = client.get(
            "/errors",
            params={
                "min_score": 0.99,
                "limit": 200,
            },
        )

    assert response.status_code == 200

    body = response.json()

    assert all(item["textcnn_score"] >= 0.99 for item in body["items"])


def test_errors_invalid_score_range():
    with TestClient(app) as client:
        response = client.get(
            "/errors",
            params={
                "min_score": 0.9,
                "max_score": 0.5,
            },
        )

    assert response.status_code == 422


def test_confusions():
    with TestClient(app) as client:
        response = client.get(
            "/errors/confusions",
            params={"limit": 5},
        )

    assert response.status_code == 200

    body = response.json()

    assert len(body["items"]) == 5

    assert all(item["count"] > 0 for item in body["items"])
