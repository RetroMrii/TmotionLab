from __future__ import annotations

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)

from backend.app.schemas.system import (
    ModelsResponse,
)


router = APIRouter(tags=["system"])


MODEL_METADATA = {
    "baseline_tfidf_dair_v001": {
        "display_name": ("TF-IDF + Logistic Regression"),
        "role": "baseline",
        "implementation": "scikit-learn",
        "capabilities": [
            "metrics",
            "held_out_compare",
        ],
    },
    "numpy_mlp_dair_v002": {
        "display_name": ("NumPy MLP"),
        "role": "educational",
        "implementation": ("manual NumPy neural network"),
        "capabilities": [
            "metrics",
            "held_out_compare",
        ],
    },
    "textcnn_dair_v001": {
        "display_name": "TextCNN",
        "role": "production",
        "implementation": "PyTorch",
        "capabilities": [
            "metrics",
            "held_out_compare",
            "live_predict",
            "live_interpret",
            "error_lab",
        ],
    },
}


@router.get(
    "/models",
    response_model=ModelsResponse,
)
def models(
    request: Request,
) -> dict:
    store = getattr(
        request.app.state,
        "artifact_store",
        None,
    )

    predictor = getattr(
        request.app.state,
        "textcnn_predictor",
        None,
    )

    if store is None:
        raise HTTPException(
            status_code=503,
            detail=("Analysis artifacts are not loaded"),
        )

    metric_rows = {item["model_id"]: item for item in store.get_metrics()}

    output = []

    for model_id, metadata in MODEL_METADATA.items():
        metrics = metric_rows[model_id]

        output.append(
            {
                "model_id": model_id,
                "display_name": (metadata["display_name"]),
                "role": metadata["role"],
                "implementation": (metadata["implementation"]),
                "runtime_loaded": (
                    predictor is not None and predictor.model_id == model_id
                ),
                "capabilities": (metadata["capabilities"]),
                "accuracy": (metrics["accuracy"]),
                "macro_f1": (metrics["macro_f1"]),
                "weighted_f1": (metrics["weighted_f1"]),
            }
        )

    return {"models": output}
