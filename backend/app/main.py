from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.app.routes.predict import (
    router as predict_router,
)
from ml.src.inference.predictor import (
    TextCNNPredictor,
)
from backend.app.routes.analysis import (
    router as analysis_router,
)
from backend.app.services.artifact_store import (
    ArtifactStore,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from backend.app.config import settings
from backend.app.errors import (
    register_exception_handlers,
)
from backend.app.routes.system import (
    router as system_router,
)

TEXTCNN_CONFIG = "ml/configs/textcnn_dair_v001.yaml"


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    predictor = TextCNNPredictor(TEXTCNN_CONFIG)

    artifact_store = ArtifactStore()

    app.state.textcnn_predictor = predictor

    app.state.artifact_store = artifact_store

    yield

    app.state.textcnn_predictor = None
    app.state.artifact_store = None


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=settings.api_description,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=False,
    allow_methods=[
        "GET",
        "POST",
        "OPTIONS",
    ],
    allow_headers=[
        "Content-Type",
        "Accept",
    ],
)

register_exception_handlers(app)


@app.get(
    "/health",
    tags=["system"],
)
@app.get(
    "/health",
    tags=["system"],
)
def health() -> dict:
    predictor = getattr(
        app.state,
        "textcnn_predictor",
        None,
    )

    artifact_store = getattr(
        app.state,
        "artifact_store",
        None,
    )

    healthy = predictor is not None and artifact_store is not None

    return {
        "status": ("ok" if healthy else "degraded"),
        "service": "TmotionLab API",
        "version": settings.api_version,
        "textcnn_loaded": (predictor is not None),
        "artifacts_loaded": (artifact_store is not None),
        "model_id": (predictor.model_id if predictor is not None else None),
    }


app.include_router(predict_router)
app.include_router(analysis_router)
app.include_router(system_router)
