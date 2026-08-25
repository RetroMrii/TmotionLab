from __future__ import annotations

from pydantic import BaseModel, Field


class ModelMetrics(BaseModel):
    model_id: str
    accuracy: float
    macro_f1: float
    weighted_f1: float


class MetricsResponse(BaseModel):
    models: list[ModelMetrics]


class CompareResponse(BaseModel):
    example_id: str
    text: str
    true_label: int
    true_label_name: str

    baseline_prediction: str
    baseline_prediction_id: int
    baseline_score: float
    baseline_correct: bool

    numpy_prediction: str
    numpy_prediction_id: int
    numpy_score: float
    numpy_correct: bool

    textcnn_prediction: str
    textcnn_prediction_id: int
    textcnn_score: float
    textcnn_correct: bool

    all_models_agree: bool
    all_models_correct: bool
    all_models_wrong: bool
    all_agree_and_correct: bool
    all_agree_and_wrong: bool

    baseline_wrong_textcnn_correct: bool
    baseline_correct_textcnn_wrong: bool
    numpy_wrong_textcnn_correct: bool
    numpy_correct_textcnn_wrong: bool

    baseline_wrong_numpy_correct: bool
    baseline_correct_numpy_wrong: bool

    three_way_disagreement: bool


class InterpretRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=5000,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=50,
    )


class InterpretationPrediction(BaseModel):
    label_id: int
    label: str
    score: float


class InterpretationScore(BaseModel):
    label: str
    score: float


class ActivationSpan(BaseModel):
    kernel_index: int
    kernel_size: int
    filter_index: int
    activation: float
    token_start: int
    token_end_exclusive: int
    span_tokens: list[str]
    span_text: str
    contains_padding: bool
    valid_text_span: bool


class InterpretResponse(BaseModel):
    model_id: str
    text: str
    tokens: list[str]
    prediction: InterpretationPrediction
    scores: list[InterpretationScore]
    top_text_spans: list[ActivationSpan]
    padding_activation_count: int
    interpretation_note: str


class ErrorItem(BaseModel):
    error_rank: int
    example_id: str
    text: str

    true_label: int
    true_label_name: str

    textcnn_prediction_id: int
    textcnn_prediction: str
    textcnn_score: float

    confusion_pair: str
    high_score_error: bool

    baseline_prediction: str
    baseline_score: float
    baseline_correct: bool

    numpy_prediction: str
    numpy_score: float
    numpy_correct: bool

    all_models_agree: bool
    all_models_wrong: bool
    all_agree_and_wrong: bool

    baseline_correct_textcnn_wrong: bool
    numpy_correct_textcnn_wrong: bool
    three_way_disagreement: bool

    interpretation_available: bool
    interpretation_cohort: str | None = None
    padding_activation_count: int | None = None

    top_activation_span_1: str | None = None
    top_activation_span_2: str | None = None
    top_activation_span_3: str | None = None


class ErrorListResponse(BaseModel):
    total: int
    offset: int
    limit: int
    items: list[ErrorItem]

    benchmark_note: str


class ConfusionPair(BaseModel):
    true_label_name: str
    textcnn_prediction: str
    confusion_pair: str
    count: int
    mean_textcnn_score: float
    max_textcnn_score: float
    high_score_errors: int


class ConfusionResponse(BaseModel):
    items: list[ConfusionPair]

    benchmark_note: str
