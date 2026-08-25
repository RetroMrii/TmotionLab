from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
    )

    include_distribution: bool = False


class EmotionProbability(BaseModel):
    label: str
    probability: float


class PredictionInput(BaseModel):
    text: str
    tokens: list[str]
    token_count: int
    truncated: bool


class PredictResponse(BaseModel):
    mode: str
    model_id: str
    top_prediction: EmotionProbability
    top_3: list[EmotionProbability]
    input: PredictionInput
    probability_note: str
    distribution: list[EmotionProbability] | None = None
