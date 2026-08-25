from __future__ import annotations

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    Request,
)

from backend.app.schemas.analysis import (
    CompareResponse,
    ConfusionResponse,
    ErrorListResponse,
    InterpretRequest,
    InterpretResponse,
    MetricsResponse,
)


router = APIRouter(tags=["analysis"])


BENCHMARK_NOTE = (
    "An evaluation error means the model prediction "
    "differs from the dataset label. It does not imply "
    "that every disagreement is an unambiguous semantic "
    "model failure."
)


def _get_store(
    request: Request,
):
    store = getattr(
        request.app.state,
        "artifact_store",
        None,
    )

    if store is None:
        raise HTTPException(
            status_code=503,
            detail=("Analysis artifacts are not loaded"),
        )

    return store


@router.get(
    "/metrics",
    response_model=MetricsResponse,
)
def metrics(
    request: Request,
) -> dict:
    store = _get_store(request)

    return {"models": (store.get_metrics())}


@router.get(
    "/compare/{example_id}",
    response_model=CompareResponse,
)
def compare_example(
    example_id: str,
    request: Request,
) -> dict:
    store = _get_store(request)

    result = store.get_comparison(example_id)

    if result is None:
        raise HTTPException(
            status_code=404,
            detail=(f"Unknown example_id: {example_id}"),
        )

    return result


@router.post(
    "/interpret",
    response_model=InterpretResponse,
)
def interpret(
    payload: InterpretRequest,
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
        return predictor.interpret(
            payload.text,
            top_k=payload.top_k,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@router.get(
    "/errors",
    response_model=ErrorListResponse,
)
def errors(
    request: Request,
    true_label: str | None = None,
    predicted_label: str | None = None,
    min_score: float = Query(
        default=0.0,
        ge=0.0,
        le=1.0,
    ),
    max_score: float = Query(
        default=1.0,
        ge=0.0,
        le=1.0,
    ),
    shared_only: bool = False,
    baseline_regression_only: bool = False,
    offset: int = Query(
        default=0,
        ge=0,
    ),
    limit: int = Query(
        default=50,
        ge=1,
        le=200,
    ),
) -> dict:
    if min_score > max_score:
        raise HTTPException(
            status_code=422,
            detail=("min_score must be <= max_score"),
        )

    store = _get_store(request)

    total, items = store.query_errors(
        true_label=true_label,
        predicted_label=predicted_label,
        min_score=min_score,
        max_score=max_score,
        shared_only=shared_only,
        baseline_regression_only=(baseline_regression_only),
        offset=offset,
        limit=limit,
    )

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "items": items,
        "benchmark_note": BENCHMARK_NOTE,
    }


@router.get(
    "/errors/confusions",
    response_model=ConfusionResponse,
)
def error_confusions(
    request: Request,
    limit: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
) -> dict:
    store = _get_store(request)

    return {
        "items": store.get_confusions(limit=limit),
        "benchmark_note": BENCHMARK_NOTE,
    }
