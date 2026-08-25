from __future__ import annotations

from pydantic import BaseModel


class ModelMetadata(BaseModel):
    model_id: str
    display_name: str
    role: str
    implementation: str
    runtime_loaded: bool
    capabilities: list[str]
    accuracy: float
    macro_f1: float
    weighted_f1: float


class ModelsResponse(BaseModel):
    models: list[ModelMetadata]
