import type {
    ApiError,
    CompareResponse,
    ConfusionResponse,
    ErrorListResponse,
    ErrorQuery,
    HealthResponse,
    InterpretResponse,
    MetricsResponse,
    ModelsResponse,
    PredictResponse,
} from "../types/api";
const API_BASE_URL =
    import.meta.env.VITE_API_BASE_URL ??
    "http://127.0.0.1:8000";

export class TmotionApiError extends Error {
    status: number;
    code: string;

    constructor(
        message: string,
        status: number,
        code: string,
    ) {
        super(message);

        this.name = "TmotionApiError";
        this.status = status;
        this.code = code;
    }
}

async function request<T>(
    path: string,
    options?: RequestInit,
): Promise<T> {
    const response = await fetch(
        `${API_BASE_URL}${path}`,
        {
            ...options,
            headers: {
                Accept: "application/json",
                ...(options?.body
                    ? {
                        "Content-Type":
                            "application/json",
                    }
                    : {}),
                ...options?.headers,
            },
        },
    );

    if (!response.ok) {
        let message =
            `Request failed with status ${response.status}`;

        let code = `http_${response.status}`;

        try {
            const payload =
                (await response.json()) as ApiError;

            if (payload.error) {
                message = payload.error.message;
                code = payload.error.code;
            }
        } catch {
            // Keep generic HTTP error.
        }

        throw new TmotionApiError(
            message,
            response.status,
            code,
        );
    }

    return (await response.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
    return request<HealthResponse>(
        "/health",
    );
}

export function getModels(): Promise<ModelsResponse> {
    return request<ModelsResponse>(
        "/models",
    );
}

export function predictText(
    text: string,
    includeDistribution = false,
): Promise<PredictResponse> {
    return request<PredictResponse>(
        "/predict",
        {
            method: "POST",
            body: JSON.stringify({
                text,
                include_distribution:
                    includeDistribution,
            }),
        },
    );
}

export function getComparison(
    exampleId: string,
): Promise<CompareResponse> {
    return request<CompareResponse>(
        `/compare/${encodeURIComponent(exampleId)}`,
    );
}

export function interpretText(
    text: string,
    topK = 10,
): Promise<InterpretResponse> {
    return request<InterpretResponse>(
        "/interpret",
        {
            method: "POST",
            body: JSON.stringify({
                text,
                top_k: topK,
            }),
        },
    );
}

export function getErrors(
    query: ErrorQuery = {},
): Promise<ErrorListResponse> {
    const params = new URLSearchParams();

    if (query.trueLabel) {
        params.set(
            "true_label",
            query.trueLabel,
        );
    }

    if (query.predictedLabel) {
        params.set(
            "predicted_label",
            query.predictedLabel,
        );
    }

    if (query.minScore !== undefined) {
        params.set(
            "min_score",
            String(query.minScore),
        );
    }

    if (query.maxScore !== undefined) {
        params.set(
            "max_score",
            String(query.maxScore),
        );
    }

    if (query.sharedOnly) {
        params.set(
            "shared_only",
            "true",
        );
    }

    if (query.baselineRegressionOnly) {
        params.set(
            "baseline_regression_only",
            "true",
        );
    }

    if (query.offset !== undefined) {
        params.set(
            "offset",
            String(query.offset),
        );
    }

    if (query.limit !== undefined) {
        params.set(
            "limit",
            String(query.limit),
        );
    }

    const search =
        params.toString();

    return request<ErrorListResponse>(
        search
            ? `/errors?${search}`
            : "/errors",
    );
}

export function getConfusions(
    limit = 10,
): Promise<ConfusionResponse> {
    return request<ConfusionResponse>(
        `/errors/confusions?limit=${limit}`,
    );
}

export function getMetrics(): Promise<MetricsResponse> {
    return request<MetricsResponse>(
        "/metrics",
    );
}