from __future__ import annotations

from fastapi import (
    APIRouter,
    HTTPException,
    Request,
)

from backend.app.schemas.prediction import (
    PredictRequest,
    PredictResponse,
)


router = APIRouter(tags=["prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(
    payload: PredictRequest,
    request: Request,
) -> dict:
    if not payload.text.strip():
        raise HTTPException(
            status_code=422,
            detail="text must not be empty",
        )

    predictor = getattr(
        request.app.state,
        "textcnn_predictor",
        None,
    )

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=("TextCNN predictor is not loaded"),
        )

    try:
        return predictor.predict(
            payload.text,
            include_distribution=(payload.include_distribution),
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
