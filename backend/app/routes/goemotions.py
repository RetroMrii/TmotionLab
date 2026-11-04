import logging
from fastapi import APIRouter, HTTPException, Request
from backend.app.schemas.goemotions import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["GoEmotions v1"])


@router.post("/v1/goemotions/predict", response_model=AnalyzeResponse)
def analyze(payload: AnalyzeRequest, request: Request):
    if not payload.text.strip():
        raise HTTPException(422, "text must not be empty")
    predictor = getattr(request.app.state, "goemotions_predictor", None)
    if predictor is None:
        raise HTTPException(503, "GoEmotions model is unavailable; see /health and server diagnostics")
    try:
        return predictor.predict(payload.text)
    except Exception as exc:
        logging.getLogger(__name__).exception("GoEmotions inference failed")
        raise HTTPException(503, "GoEmotions inference failed; see server diagnostics") from exc
