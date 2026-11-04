from __future__ import annotations

from pydantic import BaseModel


class ModelMetadata(BaseModel):
    model_id: str
    display_name: str
    role: str
    implementation: str
    runtime_loaded: bool
    capabilities: list[str]
    dataset: str
    task: str
    status: str
    prediction_endpoint: str | None = None
    threshold: float | None = None
    labels: list[str] = []
    accuracy: float | None = None
    macro_f1: float | None = None
    weighted_f1: float | None = None


class ModelsResponse(BaseModel):
    models: list[ModelMetadata]
