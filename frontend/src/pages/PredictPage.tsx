import {
    type FormEvent,
    useState,
} from "react";

import {
    predictText,
} from "../api/client";

import {
    ProbabilityBar,
} from "../components/ProbabilityBar";

import type {
    PredictResponse,
} from "../types/api";


const SAMPLE_TEXT =
    "I feel extremely nervous about tomorrow.";


export function PredictPage() {
    const [
        text,
        setText,
    ] = useState(SAMPLE_TEXT);

    const [
        result,
        setResult,
    ] = useState<PredictResponse | null>(
        null,
    );

    const [
        loading,
        setLoading,
    ] = useState(false);

    const [
        distributionLoading,
        setDistributionLoading,
    ] = useState(false);

    const [
        error,
        setError,
    ] = useState<string | null>(
        null,
    );

    async function handleSubmit(
        event: FormEvent,
    ) {
        event.preventDefault();

        if (!text.trim()) {
            setError(
                "Enter text before running prediction.",
            );

            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await predictText(
                text,
                false,
            );

            setResult(response);
        } catch (caught) {
            setResult(null);

            setError(
                caught instanceof Error
                    ? caught.message
                    : "Prediction request failed",
            );
        } finally {
            setLoading(false);
        }
    }

    async function loadFullDistribution() {
        if (!result) {
            return;
        }

        if (result.distribution) {
            setResult({
                ...result,
                distribution: null,
            });

            return;
        }

        setDistributionLoading(true);
        setError(null);

        try {
            const response = await predictText(
                result.input.text,
                true,
            );

            setResult(response);
        } catch (caught) {
            setError(
                caught instanceof Error
                    ? caught.message
                    : "Distribution request failed",
            );
        } finally {
            setDistributionLoading(false);
        }
    }

    return (
        <div className="page">
            <section className="page-heading">
                <div className="eyebrow">
                    Predict
                </div>

                <h1>
                    TextCNN emotion inference
                </h1>

                <p>
                    Submit English text to the promoted
                    TextCNN model. The default view shows
                    the three highest softmax probabilities.
                </p>
            </section>

            <div className="predict-layout">
                <section className="panel">
                    <form
                        onSubmit={handleSubmit}
                    >
                        <label
                            className="field-label"
                            htmlFor="prediction-text"
                        >
                            Input text
                        </label>

                        <textarea
                            id="prediction-text"
                            className="text-input"
                            value={text}
                            onChange={(event) =>
                                setText(
                                    event.target.value,
                                )
                            }
                            rows={9}
                            maxLength={5000}
                            placeholder="Enter English text..."
                        />

                        <div className="input-footer">
                            <span className="muted">
                                {text.length} / 5000
                            </span>

                            <button
                                className="primary-button"
                                type="submit"
                                disabled={loading}
                            >
                                {loading
                                    ? "Running..."
                                    : "Run TextCNN"}
                            </button>
                        </div>
                    </form>

                    {error && (
                        <div className="alert alert-error">
                            {error}
                        </div>
                    )}
                </section>

                <section className="panel result-panel">
                    {!result && (
                        <div className="empty-state">
                            <div className="empty-state-title">
                                No prediction yet
                            </div>

                            <p>
                                Run the model to inspect its
                                top prediction and probability
                                distribution.
                            </p>
                        </div>
                    )}

                    {result && (
                        <>
                            <div className="result-header">
                                <div>
                                    <div className="eyebrow">
                                        Top prediction
                                    </div>

                                    <h2 className="prediction-label">
                                        {
                                            result.top_prediction
                                                .label
                                        }
                                    </h2>
                                </div>

                                <div className="prediction-score">
                                    {(
                                        result.top_prediction
                                            .probability * 100
                                    ).toFixed(2)}
                                    %
                                </div>
                            </div>

                            <div className="section-divider" />

                            <h3>
                                Top 3
                            </h3>

                            <div className="probability-list">
                                {result.top_3.map(
                                    (item) => (
                                        <ProbabilityBar
                                            key={item.label}
                                            item={item}
                                        />
                                    ),
                                )}
                            </div>

                            <button
                                type="button"
                                className="secondary-button"
                                onClick={
                                    loadFullDistribution
                                }
                                disabled={
                                    distributionLoading
                                }
                            >
                                {distributionLoading
                                    ? "Loading..."
                                    : result.distribution
                                        ? "Hide full distribution"
                                        : "Show full distribution"}
                            </button>

                            {result.distribution && (
                                <div className="distribution-panel">
                                    <h3>
                                        Full distribution
                                    </h3>

                                    <div className="probability-list">
                                        {result.distribution.map(
                                            (item) => (
                                                <ProbabilityBar
                                                    key={item.label}
                                                    item={item}
                                                />
                                            ),
                                        )}
                                    </div>
                                </div>
                            )}

                            <div className="model-note">
                                <strong>
                                    Model:
                                </strong>
                                {" "}
                                {result.model_id}

                                <br />

                                <strong>
                                    Tokens:
                                </strong>
                                {" "}
                                {result.input.token_count}

                                {result.input.truncated && (
                                    <>
                                        <br />
                                        Input was truncated to the
                                        model sequence limit.
                                    </>
                                )}
                            </div>

                            <div className="alert alert-info">
                                {
                                    result.probability_note
                                }
                            </div>
                        </>
                    )}
                </section>
            </div>
        </div>
    );
}