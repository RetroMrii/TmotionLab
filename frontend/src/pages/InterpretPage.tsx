import {
    type FormEvent,
    useMemo,
    useState,
} from "react";

import {
    interpretText,
} from "../api/client";

import {
    ActivationSpanCard,
} from "../components/ActivationSpanCard";

import type {
    ActivationSpan,
    InterpretResponse,
} from "../types/api";


const SAMPLE_TEXT =
    "I feel extremely nervous about tomorrow but also a little excited.";


function tokenIsHighlighted(
    tokenIndex: number,
    spans: ActivationSpan[],
): boolean {
    return spans.some(
        (span) =>
            tokenIndex >= span.token_start
            && tokenIndex < span.token_end_exclusive,
    );
}


export function InterpretPage() {
    const [
        text,
        setText,
    ] = useState(SAMPLE_TEXT);

    const [
        topK,
        setTopK,
    ] = useState(10);

    const [
        result,
        setResult,
    ] = useState<InterpretResponse | null>(
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

    const highlightedTokens = useMemo(() => {
        if (!result) {
            return [];
        }

        return result.tokens.map(
            (token, index) => ({
                token,
                highlighted:
                    tokenIsHighlighted(
                        index,
                        result.top_text_spans,
                    ),
            }),
        );
    }, [result]);

    async function handleSubmit(
        event: FormEvent,
    ) {
        event.preventDefault();

        if (!text.trim()) {
            setError(
                "Enter text before running interpretation.",
            );

            return;
        }

        setLoading(true);
        setError(null);

        try {
            const response = await interpretText(
                text,
                topK,
            );

            setResult(response);
        } catch (caught) {
            setResult(null);

            setError(
                caught instanceof Error
                    ? caught.message
                    : "Interpretation request failed",
            );
        } finally {
            setLoading(false);
        }
    }

    return (
        <div className="page">
            <section className="page-heading">
                <div className="eyebrow">
                    Interpret
                </div>

                <h1>
                    Inspect TextCNN activation regions
                </h1>

                <p>
                    Trace strong convolutional activations
                    back to exact input token spans.
                    These spans expose internal model
                    responses, not causal explanations of
                    emotion.
                </p>
            </section>

            <div className="interpret-layout">
                <section className="panel">
                    <form onSubmit={handleSubmit}>
                        <label
                            className="field-label"
                            htmlFor="interpret-text"
                        >
                            Input text
                        </label>

                        <textarea
                            id="interpret-text"
                            className="text-input"
                            value={text}
                            onChange={(event) =>
                                setText(
                                    event.target.value,
                                )
                            }
                            rows={8}
                            maxLength={5000}
                        />

                        <div className="interpret-controls">
                            <div>
                                <label
                                    className="field-label"
                                    htmlFor="top-k"
                                >
                                    Top activation spans
                                </label>

                                <select
                                    id="top-k"
                                    className="select-field"
                                    value={topK}
                                    onChange={(event) =>
                                        setTopK(
                                            Number(
                                                event.target.value,
                                            ),
                                        )
                                    }
                                >
                                    <option value={5}>
                                        5
                                    </option>
                                    <option value={10}>
                                        10
                                    </option>
                                    <option value={15}>
                                        15
                                    </option>
                                    <option value={20}>
                                        20
                                    </option>
                                </select>
                            </div>

                            <button
                                className="primary-button"
                                type="submit"
                                disabled={loading}
                            >
                                {loading
                                    ? "Inspecting..."
                                    : "Run interpretation"}
                            </button>
                        </div>
                    </form>

                    {error && (
                        <div className="alert alert-error">
                            {error}
                        </div>
                    )}
                </section>

                <section className="panel interpret-summary-panel">
                    {!result && (
                        <div className="empty-state">
                            <div className="empty-state-title">
                                No interpretation yet
                            </div>

                            <p>
                                Run the TextCNN interpreter to
                                inspect its strongest valid
                                convolutional activation spans.
                            </p>
                        </div>
                    )}

                    {result && (
                        <>
                            <div className="result-header">
                                <div>
                                    <div className="eyebrow">
                                        Prediction
                                    </div>

                                    <h2 className="prediction-label">
                                        {
                                            result.prediction
                                                .label
                                        }
                                    </h2>
                                </div>

                                <div className="prediction-score">
                                    {(
                                        result.prediction
                                            .score * 100
                                    ).toFixed(2)}
                                    %
                                </div>
                            </div>

                            <div className="model-note">
                                <strong>
                                    Model:
                                </strong>
                                {" "}
                                {result.model_id}
                            </div>

                            <div className="section-divider" />

                            <div className="interpret-stat-grid">
                                <div>
                                    <span className="metric-name">
                                        Tokens
                                    </span>

                                    <strong>
                                        {result.tokens.length}
                                    </strong>
                                </div>

                                <div>
                                    <span className="metric-name">
                                        Valid top spans
                                    </span>

                                    <strong>
                                        {
                                            result.top_text_spans
                                                .length
                                        }
                                    </strong>
                                </div>

                                <div>
                                    <span className="metric-name">
                                        Padding-overlap maxima
                                    </span>

                                    <strong>
                                        {
                                            result
                                                .padding_activation_count
                                        }
                                    </strong>
                                </div>
                            </div>

                            <div className="alert alert-info">
                                {result.interpretation_note}
                            </div>
                        </>
                    )}
                </section>
            </div>

            {result && (
                <>
                    <section className="panel token-panel">
                        <div className="section-header">
                            <div>
                                <div className="eyebrow">
                                    Token map
                                </div>

                                <h2>
                                    Activated text regions
                                </h2>
                            </div>
                        </div>

                        <div className="token-strip">
                            {highlightedTokens.map(
                                (
                                    {
                                        token,
                                        highlighted,
                                    },
                                    index,
                                ) => (
                                    <span
                                        className={
                                            highlighted
                                                ? "token token-active"
                                                : "token"
                                        }
                                        key={`${index}-${token}`}
                                    >
                                        <span className="token-index">
                                            {index}
                                        </span>

                                        {token}
                                    </span>
                                ),
                            )}
                        </div>

                        <div className="comparison-score-note">
                            Highlighting means the token
                            participates in at least one
                            displayed top activation span.
                            Overlapping spans are collapsed
                            visually in this token map.
                        </div>
                    </section>

                    <section className="interpret-results-section">
                        <div className="section-header">
                            <div>
                                <div className="eyebrow">
                                    Activation evidence
                                </div>

                                <h2>
                                    Top convolutional spans
                                </h2>
                            </div>
                        </div>

                        <div className="activation-grid">
                            {result.top_text_spans.map(
                                (span, index) => (
                                    <ActivationSpanCard
                                        key={
                                            `${span.kernel_size}-`
                                            + `${span.filter_index}-`
                                            + `${span.token_start}`
                                        }
                                        rank={index + 1}
                                        span={span}
                                    />
                                ),
                            )}
                        </div>
                    </section>

                    <section className="panel interpretation-caveat-panel">
                        <div className="eyebrow">
                            Interpretation caveat
                        </div>

                        <h2>
                            What these spans do—and do not—mean
                        </h2>

                        <div className="interpretation-rule-grid">
                            <div>
                                <h3>
                                    Supported
                                </h3>

                                <p>
                                    These local token regions
                                    produced strong post-ReLU
                                    convolutional activations in
                                    the trained TextCNN.
                                </p>
                            </div>

                            <div>
                                <h3>
                                    Not supported
                                </h3>

                                <p>
                                    The spans are not proof that
                                    these words caused the final
                                    class prediction or represent
                                    the writer&apos;s true emotional
                                    state.
                                </p>
                            </div>
                        </div>
                    </section>
                </>
            )}
        </div>
    );
}