from fastapi import APIRouter, Request
from backend.app.schemas.system import ModelsResponse
from backend.app.services.goemotions import MODEL_ID, THRESHOLD

router = APIRouter(tags=["system"])
HISTORICAL = {
    "baseline_tfidf_dair_v001": ("TF-IDF + Logistic Regression", "scikit-learn"),
    "numpy_mlp_dair_v002": ("NumPy MLP", "manual NumPy neural network"),
    "textcnn_dair_v001": ("TextCNN", "PyTorch"),
}


@router.get("/models", response_model=ModelsResponse)
def models(request: Request):
    predictor = getattr(request.app.state, "goemotions_predictor", None)
    store = getattr(request.app.state, "artifact_store", None)
    rows = {row["model_id"]: row for row in store.get_metrics()} if store else {}
    output = [{
        "model_id": MODEL_ID, "display_name": "GoEmotions RoBERTa (real rare-label 4x, seed 42)",
        "role": "production", "implementation": "Transformers / PyTorch",
        "runtime_loaded": predictor is not None,
        "status": "available" if predictor else "unavailable",
        "dataset": "goemotions_v001", "task": "multi_label",
        "capabilities": ["live_predict"] if predictor else [],
        "prediction_endpoint": "/v1/goemotions/predict", "threshold": THRESHOLD,
        "labels": predictor.labels if predictor else [],
    }]
    for model_id, (name, implementation) in HISTORICAL.items():
        capabilities = ["metrics", "held_out_compare"] if store else []
        if store and model_id == "textcnn_dair_v001":
            capabilities.append("error_lab")
        output.append({
            "model_id": model_id, "display_name": name, "implementation": implementation,
            "role": "historical", "runtime_loaded": False,
            "status": "offline_artifacts" if store else "unavailable",
            "dataset": "dair_ai_emotion", "task": "single_label",
            "capabilities": capabilities,
            "labels": ["sadness", "joy", "love", "anger", "fear", "surprise"],
            **rows.get(model_id, {}),
        })
    return {"models": output}
