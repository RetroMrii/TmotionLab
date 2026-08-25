import {
    type FormEvent,
    useState,
} from "react";

import {
    getComparison,
} from "../api/client";

import {
    ModelPredictionCard,
} from "./ModelPredictionCard";

import type {
    CompareResponse,
} from "../types/api";


const SAMPLE_IDS = [
    "dair_test_001837",
    "dair_test_001943",
    "dair_test_000353",
    "dair_test_000265",
    "dair_test_000576",
];


export function ComparePage() {
    const [
        exampleId,
        setExampleId,
    ] = useState(
        "dair_test_001837",
    );

    const [
        result,
        setResult,
    ] = useState<CompareResponse | null>(
        null,
    );

    const [
        loading,
        setLoading,
    ] = useState(false);

    const [
        error,
        setError,
    ] = useState<string | null>(
        null,
    );

    async function loadExample(
        id: string,
    ) {
        const normalized = id.trim();

        if (!normalized) {
            setError(
                "Enter a test example ID.",
            );

            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await getComparison(
                normalized,
            );

            setResult(response);
            setExampleId(normalized);
        } catch (caught) {
            setResult(null);

            setError(
                caught instanceof Error
                    ? caught.message
                    : "Comparison request failed",
            );
        } finally {
            setLoading(false);
        }
    }

    async function handleSubmit(
        event: FormEvent,
    ) {
        event.preventDefault();

        await loadExample(
            exampleId,
        );
    }

    function getAgreementLabel(
        data: CompareResponse,
    ): string {
        if (
            data.all_models_agree
            && data.all_models_correct
        ) {
            return "All three models agree with the benchmark";
        }

        if (data.all_agree_and_wrong) {
            return "All three models agree on the same alternative label";
        }

        if (data.three_way_disagreement) {
            return "Three-way disagreement";
        }

        if (data.all_models_agree) {
            return "All three models agree";
        }

        return "Partial model disagreement";
    }

    return (
        <div className="page">
            <section className="page-heading">
                <div className="eyebrow">
                    Compare
                </div>

                <h1>
                    Same example, three model families
                </h1>

                <p>
                    Compare the saved held-out predictions
                    from Logistic Regression, the NumPy MLP,
                    and TextCNN on exactly the same test
                    example.
                </p>
            </section>

            <section className="panel compare-search-panel">
                <form
                    className="compare-search-form"
                    onSubmit={handleSubmit}
                >
                    <div className="compare-input-block">
                        <label
                            className="field-label"
                            htmlFor="example-id"
                        >
                            Test example ID
                        </label>

                        <input
                            id="example-id"
                            className="text-field"
                            value={exampleId}
                            onChange={(event) =>
                                setExampleId(
                                    event.target.value,
                                )
                            }
                            placeholder="dair_test_001837"
                        />
                    </div>

                    <button
                        className="primary-button"
                        type="submit"
                        disabled={loading}
                    >
                        {loading
                            ? "Loading..."
                            : "Load comparison"}
                    </button>
                </form>

                <div className="sample-id-row">
                    <span className="muted">
                        Example cases:
                    </span>

                    {SAMPLE_IDS.map(
                        (id) => (
                            <button
                                type="button"
                                className="example-chip"
                                key={id}
                                onClick={() => {
                                    void loadExample(id);
                                }}
                            >
                                {id}
                            </button>
                        ),
                    )}
                </div>

                {error && (
                    <div className="alert alert-error">
                        {error}
                    </div>
                )}
            </section>

            {!result && (
                <section className="panel compare-empty-panel">
                    <div className="empty-state">
                        <div className="empty-state-title">
                            Load a held-out example
                        </div>

                        <p>
                            Comparison is based on stable test
                            IDs so every model is evaluated on
                            the same text and benchmark label.
                        </p>
                    </div>
                </section>
            )}

            {result && (
                <>
                    <section className="comparison-example">
                        <div className="comparison-meta-row">
                            <div>
                                <div className="eyebrow">
                                    Example
                                </div>

                                <h2>
                                    {result.example_id}
                                </h2>
                            </div>

                            <div className="truth-badge">
                                Benchmark label:
                                {" "}
                                <strong>
                                    {result.true_label_name}
                                </strong>
                            </div>
                        </div>

                        <blockquote className="comparison-text">
                            {result.text}
                        </blockquote>
                    </section>

                    <section className="comparison-model-grid">
                        <ModelPredictionCard
                            name="TF-IDF + Logistic Regression"
                            role="baseline"
                            prediction={
                                result.baseline_prediction
                            }
                            score={
                                result.baseline_score
                            }
                            correct={
                                result.baseline_correct
                            }
                        />

                        <ModelPredictionCard
                            name="NumPy MLP"
                            role="educational"
                            prediction={
                                result.numpy_prediction
                            }
                            score={
                                result.numpy_score
                            }
                            correct={
                                result.numpy_correct
                            }
                        />

                        <ModelPredictionCard
                            name="TextCNN"
                            role="production"
                            prediction={
                                result.textcnn_prediction
                            }
                            score={
                                result.textcnn_score
                            }
                            correct={
                                result.textcnn_correct
                            }
                        />
                    </section>

                    <section className="panel comparison-analysis-panel">
                        <div className="section-header">
                            <div>
                                <div className="eyebrow">
                                    Behavior
                                </div>

                                <h2>
                                    Agreement analysis
                                </h2>
                            </div>

                            <div className="agreement-badge">
                                {getAgreementLabel(result)}
                            </div>
                        </div>

                        <div className="comparison-signals">
                            <div className="signal-row">
                                <span>
                                    All models agree
                                </span>

                                <strong>
                                    {result.all_models_agree
                                        ? "Yes"
                                        : "No"}
                                </strong>
                            </div>

                            <div className="signal-row">
                                <span>
                                    All match benchmark
                                </span>

                                <strong>
                                    {result.all_models_correct
                                        ? "Yes"
                                        : "No"}
                                </strong>
                            </div>

                            <div className="signal-row">
                                <span>
                                    All disagree with benchmark
                                </span>

                                <strong>
                                    {result.all_models_wrong
                                        ? "Yes"
                                        : "No"}
                                </strong>
                            </div>

                            <div className="signal-row">
                                <span>
                                    Three different predictions
                                </span>

                                <strong>
                                    {result.three_way_disagreement
                                        ? "Yes"
                                        : "No"}
                                </strong>
                            </div>
                        </div>

                        {result.all_agree_and_wrong && (
                            <div className="alert alert-info">
                                All three model families choose
                                the same alternative label.
                                This is still an evaluation error
                                relative to the dataset, but it
                                does not prove the model prediction
                                is semantically unreasonable.
                            </div>
                        )}

                        {result.baseline_correct_textcnn_wrong && (
                            <div className="alert alert-warning">
                                This is a TextCNN regression:
                                the baseline matches the benchmark
                                while TextCNN does not.
                            </div>
                        )}

                        {result.baseline_wrong_textcnn_correct && (
                            <div className="alert alert-success">
                                TextCNN repairs a baseline
                                benchmark disagreement on this
                                example.
                            </div>
                        )}

                        <div className="comparison-score-note">
                            Raw scores from different model
                            families are shown for inspection
                            only. They are not assumed to be
                            calibrated or directly comparable.
                        </div>
                    </section>
                </>
            )}
        </div>
    );
}