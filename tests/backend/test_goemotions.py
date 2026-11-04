import json
from concurrent.futures import ThreadPoolExecutor
from itertools import islice

import pytest
from fastapi.testclient import TestClient
import backend.app.main as main
from backend.app.services.goemotions import (
    FREEZE_PATH, MANIFEST_PATH, MODEL_DIR, MODEL_ID, ROOT,
    score_payload, verify_artifacts,
)

ROUTE = "/v1/goemotions/predict"


@pytest.mark.parametrize("indices", [[], [0], [0, 18, 27]])
def test_threshold_sets_including_neutral(indices, real_predictor, monkeypatch):
    values = [0.2499] * 28
    for index in indices:
        values[index] = 0.25
    body = real_predictor.predict("A standalone contract fixture.")
    body.update(score_payload(real_predictor.labels, values))
    monkeypatch.setattr(real_predictor, "predict", lambda text: {**body, "input": {**body["input"], "text": text}})
    with TestClient(main.app) as client:
        response = client.post(ROUTE, json={"text": "Contract fixture"})
    assert response.status_code == 200
    result = response.json()
    assert result["detected_labels"] == [real_predictor.labels[i] for i in indices]
    assert len(result["ranked_scores"]) == 5
    assert [s["label_id"] for s in result["scores"]] == list(range(28))
    assert result["ranked_scores"] == sorted(result["scores"], key=lambda s: (-s["score"], s["label_id"]))[:5]
    assert "top_prediction" not in result


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "   "}, {"text": "a" * 5001}, {"text": 12}, {"text": "Hi", "threshold": 0.1}])
def test_validation(payload):
    with TestClient(main.app) as client:
        response = client.post(ROUTE, json=payload)
    assert response.status_code == 422
    assert "error" in response.json()


def test_real_multilabel_and_exact_input(real_predictor):
    text = "  I love this, thank you so much!  "
    with TestClient(main.app) as client:
        result = client.post(ROUTE, json={"text": text}).json()
    assert result["model_id"] == MODEL_ID
    assert result["input"]["text"] == text
    assert {"love", "gratitude"}.issubset(result["detected_labels"])
    assert sum(item["score"] for item in result["scores"]) > 1.1
    assert result["threshold"] == 0.25
    assert result["input"]["truncated"] is False


def test_long_input(real_predictor):
    result = real_predictor.predict("wonderful " * 500)
    assert result["input"]["truncated"] is True
    assert result["input"]["token_count"] == 48
    assert len(result["input"]["tokens"]) == 48
    assert result["input"]["original_token_count"] > 48


def test_validation_fixture_compatibility(real_predictor):
    # Read only validation rows; never consult held-out test examples.
    with (ROOT / "data/processed/goemotions_v001/validation.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in islice(stream, 16)]
    path = ROOT / f"ml/artifacts/predictions/{MODEL_ID}_validation.jsonl"
    with path.open(encoding="utf-8") as stream:
        expected = [json.loads(line) for line in islice(stream, 16)]
    max_difference = 0
    for source, saved in zip(rows, expected, strict=True):
        assert source["example_id"] == saved["example_id"]
        actual = real_predictor.predict(source["text"])
        differences = [abs(s["score"] - p) for s, p in zip(actual["scores"], saved["probabilities"], strict=True)]
        max_difference = max(max_difference, max(differences))
    print(f"validation sample=16 maximum probability difference={max_difference:.9f}")
    assert max_difference < 0.002


def test_concurrent_predictions_are_reproducible(real_predictor):
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(real_predictor.predict, ["Thank you for your kindness."] * 3))
    assert results[0] == results[1] == results[2]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -0.1, 1.1])
def test_nonfinite_or_invalid_scores_rejected(bad, real_predictor):
    with pytest.raises(ValueError):
        score_payload(real_predictor.labels, [bad] * 28)


def test_inference_failure_returns_json_503(monkeypatch, real_predictor):
    def fail(_):
        raise ValueError("Invalid classifier logits")
    monkeypatch.setattr(real_predictor, "predict", fail)
    with TestClient(main.app) as client:
        response = client.post(ROUTE, json={"text": "hello"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "http_503"


@pytest.mark.parametrize("value,shape,status", [(0.0, (1, 28), 200), (float("nan"), (1, 28), 503), (float("inf"), (1, 28), 503), (0.0, (1, 6), 503)])
def test_logits_shape_finiteness_and_sigmoid_once(value, shape, status, monkeypatch, real_predictor):
    import torch
    from types import SimpleNamespace
    monkeypatch.setattr(real_predictor.model, "forward", lambda **kwargs: SimpleNamespace(logits=torch.full(shape, value)))
    with TestClient(main.app) as client:
        response = client.post(ROUTE, json={"text": "Independent sigmoid check"})
    assert response.status_code == status
    if status == 200:
        assert all(item["score"] == 0.5 for item in response.json()["scores"])
        assert response.json()["detected_labels"] == real_predictor.labels


def test_missing_or_changed_freeze_metadata_fails_closed(tmp_path):
    freeze = json.loads(FREEZE_PATH.read_text())
    path = tmp_path / "freeze.json"
    freeze["classification_threshold"] = 0.5
    path.write_text(json.dumps(freeze))
    with pytest.raises(ValueError, match="provenance"):
        verify_artifacts(freeze_path=path)
    del freeze["seed"]
    path.write_text(json.dumps(freeze))
    with pytest.raises(KeyError):
        verify_artifacts(freeze_path=path)


def test_missing_model_does_not_block_offline_tools(monkeypatch):
    def fail():
        raise FileNotFoundError("missing local weights")
    monkeypatch.setattr(main, "GoEmotionsPredictor", fail)
    with TestClient(main.app) as client:
        assert client.get("/health").json()["status"] == "degraded"
        model = client.get("/models").json()["models"][0]
        assert model["capabilities"] == []
        assert model["runtime_loaded"] is False
        assert client.post(ROUTE, json={"text": "hello"}).status_code == 503
        assert client.get("/metrics").status_code == 200


def test_missing_archive_does_not_block_analyze(monkeypatch):
    def fail():
        raise FileNotFoundError("archive absent")
    monkeypatch.setattr(main, "ArtifactStore", fail)
    with TestClient(main.app) as client:
        assert client.post(ROUTE, json={"text": "hello"}).status_code == 200
        assert client.get("/health").json()["artifacts_loaded"] is False
        assert client.get("/metrics").status_code == 503
        assert all(not model["capabilities"] for model in client.get("/models").json()["models"][1:])


def test_freeze_hash_and_mapping_fail_closed(tmp_path, monkeypatch):
    import backend.app.services.goemotions as service
    labels = verify_artifacts()
    freeze = json.loads(FREEZE_PATH.read_text())
    # Virtual small copy: exercise checks without duplicating half a GB of weights.
    for name in freeze["model_file_hashes"]:
        (tmp_path / name).write_text("corrupted")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_artifacts(model_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_artifacts(model_dir=tmp_path / "missing")
    monkeypatch.setattr(service, "sha256", lambda path: freeze["dataset_manifest_sha256"]
                        if path == MANIFEST_PATH else freeze["model_file_hashes"][path.name])
    config = json.loads((MODEL_DIR / "config.json").read_text())
    config["id2label"]["0"] = labels[1]
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mapping"):
        verify_artifacts(model_dir=tmp_path)


def test_paths_do_not_depend_on_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert len(verify_artifacts()) == 28
    assert main.ArtifactStore().query_errors()[0] == 207


def test_routes_remain_discoverable_without_legacy_imports():
    assert not (ROOT / "ml/src/inference/predictor.py").exists()
    with TestClient(main.app) as client:
        paths = client.get("/openapi.json").json()["paths"]
    assert {ROUTE, "/predict", "/interpret", "/health", "/models", "/metrics", "/compare/{example_id}", "/errors", "/errors/confusions"} <= paths.keys()
