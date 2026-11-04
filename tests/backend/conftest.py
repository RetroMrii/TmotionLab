import pytest
import backend.app.main as main
from backend.app.services.goemotions import GoEmotionsPredictor


@pytest.fixture(scope="session")
def real_predictor():
    return GoEmotionsPredictor()


@pytest.fixture(autouse=True)
def reuse_loaded_model(monkeypatch, real_predictor):
    # Real local weights, loaded once for the regression session, not per client.
    monkeypatch.setattr(main, "GoEmotionsPredictor", lambda: real_predictor)
