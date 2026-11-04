from typing import Literal
from pydantic import BaseModel, ConfigDict, Field


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1, max_length=5000, strict=True)


class LabelScore(BaseModel):
    label_id: int = Field(ge=0, le=27)
    label: str
    score: float = Field(ge=0, le=1, allow_inf_nan=False)


class AnalyzedInput(BaseModel):
    text: str
    tokens: list[str]
    token_count: int
    original_token_count: int
    max_tokens: Literal[48]
    truncated: bool


class AnalyzeResponse(BaseModel):
    schema_version: Literal["1"]
    task: Literal["multi_label"]
    model_id: Literal["roberta_goemotions_rare4x_v001"]
    dataset: Literal["goemotions_v001"]
    seed: Literal[42]
    threshold: Literal[0.25]
    score_type: Literal["sigmoid"]
    device: str
    input: AnalyzedInput
    scores: list[LabelScore] = Field(min_length=28, max_length=28)
    detected_labels: list[str]
    ranked_scores: list[LabelScore] = Field(min_length=5, max_length=5)
    score_note: str
    availability_notes: list[str]
