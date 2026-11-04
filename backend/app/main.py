from __future__ import annotations
from contextlib import asynccontextmanager
from fastapi import FastAPI
from backend.app.routes.predict import (
    router as predict_router,
)
import logging
from backend.app.services.goemotions import GoEmotionsPredictor
from backend.app.routes.goemotions import router as goemotions_router
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.textcnn_predictor = None
    app.state.goemotions_predictor = None
    app.state.artifact_store = None
    app.state.diagnostics = {"legacy_runtime": "Archived DAIR runtime is unavailable; /predict and /interpret return 503."}
    for key, factory in (("goemotions_predictor", GoEmotionsPredictor), ("artifact_store", ArtifactStore)):
        try:
            setattr(app.state, key, factory())
        except Exception:
            logging.getLogger(__name__).exception("Unable to initialize %s", key)
            app.state.diagnostics[key] = "Unavailable: initialization failed; see server logs."
    try:
        yield
    finally:
        app.state.goemotions_predictor = None
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


@app.get("/health", tags=["system"])
def health() -> dict:
    predictor = getattr(app.state, "goemotions_predictor", None)
    store = getattr(app.state, "artifact_store", None)
    return {
        "status": "ok" if predictor is not None else "degraded",
        "service": "TmotionLab API", "version": settings.api_version,
        "goemotions_loaded": predictor is not None,
        "textcnn_loaded": False, "artifacts_loaded": store is not None,
        "model_id": predictor.model_id if predictor else None,
        "diagnostics": getattr(app.state, "diagnostics", {}),
    }


app.include_router(predict_router)
app.include_router(analysis_router)
app.include_router(system_router)

app.include_router(goemotions_router)
