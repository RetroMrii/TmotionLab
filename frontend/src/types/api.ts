export interface ApiError {
    error: {
        code: string;
        message: string;
        details: unknown;
    };
}

export interface HealthResponse {
    status: "ok" | "degraded";
    service: string;
    version: string;
    textcnn_loaded: boolean;
    artifacts_loaded: boolean;
    model_id: string | null;
}

export interface ModelMetadata {
    model_id: string;
    display_name: string;
    role: string;
    implementation: string;
    runtime_loaded: boolean;
    capabilities: string[];
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
}

export interface ModelsResponse {
    models: ModelMetadata[];
}

export interface EmotionProbability {
    label: string;
    probability: number;
}

export interface PredictionInput {
    text: string;
    tokens: string[];
    token_count: number;
    truncated: boolean;
}

export interface PredictResponse {
    mode: string;
    model_id: string;
    top_prediction: EmotionProbability;
    top_3: EmotionProbability[];
    input: PredictionInput;
    probability_note: string;
    distribution: EmotionProbability[] | null;
}

export interface CompareResponse {
    example_id: string;
    text: string;

    true_label: number;
    true_label_name: string;

    baseline_prediction: string;
    baseline_prediction_id: number;
    baseline_score: number;
    baseline_correct: boolean;

    numpy_prediction: string;
    numpy_prediction_id: number;
    numpy_score: number;
    numpy_correct: boolean;

    textcnn_prediction: string;
    textcnn_prediction_id: number;
    textcnn_score: number;
    textcnn_correct: boolean;

    all_models_agree: boolean;
    all_models_correct: boolean;
    all_models_wrong: boolean;
    all_agree_and_correct: boolean;
    all_agree_and_wrong: boolean;

    baseline_wrong_textcnn_correct: boolean;
    baseline_correct_textcnn_wrong: boolean;

    numpy_wrong_textcnn_correct: boolean;
    numpy_correct_textcnn_wrong: boolean;

    baseline_wrong_numpy_correct: boolean;
    baseline_correct_numpy_wrong: boolean;

    three_way_disagreement: boolean;
}

export interface InterpretationPrediction {
    label_id: number;
    label: string;
    score: number;
}

export interface InterpretationScore {
    label: string;
    score: number;
}

export interface ActivationSpan {
    kernel_index: number;
    kernel_size: number;
    filter_index: number;
    activation: number;
    token_start: number;
    token_end_exclusive: number;
    span_tokens: string[];
    span_text: string;
    contains_padding: boolean;
    valid_text_span: boolean;
}

export interface InterpretResponse {
    model_id: string;
    text: string;
    tokens: string[];
    prediction: InterpretationPrediction;
    scores: InterpretationScore[];
    top_text_spans: ActivationSpan[];
    padding_activation_count: number;
    interpretation_note: string;
}

export interface ErrorItem {
    error_rank: number;
    example_id: string;
    text: string;

    true_label: number;
    true_label_name: string;

    textcnn_prediction_id: number;
    textcnn_prediction: string;
    textcnn_score: number;

    confusion_pair: string;
    high_score_error: boolean;

    baseline_prediction: string;
    baseline_score: number;
    baseline_correct: boolean;

    numpy_prediction: string;
    numpy_score: number;
    numpy_correct: boolean;

    all_models_agree: boolean;
    all_models_wrong: boolean;
    all_agree_and_wrong: boolean;

    baseline_correct_textcnn_wrong: boolean;
    numpy_correct_textcnn_wrong: boolean;
    three_way_disagreement: boolean;

    interpretation_available: boolean;
    interpretation_cohort: string | null;
    padding_activation_count: number | null;

    top_activation_span_1: string | null;
    top_activation_span_2: string | null;
    top_activation_span_3: string | null;
}

export interface ErrorListResponse {
    total: number;
    offset: number;
    limit: number;
    items: ErrorItem[];
    benchmark_note: string;
}

export interface ConfusionPair {
    true_label_name: string;
    textcnn_prediction: string;
    confusion_pair: string;
    count: number;
    mean_textcnn_score: number;
    max_textcnn_score: number;
    high_score_errors: number;
}

export interface ConfusionResponse {
    items: ConfusionPair[];
    benchmark_note: string;
}

export interface ErrorQuery {
    trueLabel?: string;
    predictedLabel?: string;
    minScore?: number;
    maxScore?: number;
    sharedOnly?: boolean;
    baselineRegressionOnly?: boolean;
    offset?: number;
    limit?: number;
}

export interface ModelMetrics {
    model_id: string;
    accuracy: number;
    macro_f1: number;
    weighted_f1: number;
}

export interface MetricsResponse {
    models: ModelMetrics[];
}