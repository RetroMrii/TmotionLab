import {
    type FormEvent,
    useEffect,
    useState,
} from "react";

import {
    getConfusions,
    getErrors,
} from "../api/client";

import {
    ConfusionTable,
} from "../components/ConfusionTable";

import {
    ErrorCard,
} from "../components/ErrorCard";

import type {
    ConfusionPair,
    ErrorItem,
} from "../types/api";


const LABELS = [
    "",
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
];

const PAGE_SIZE = 20;


export function ErrorLabPage() {
    const [
        trueLabel,
        setTrueLabel,
    ] = useState("");

    const [
        predictedLabel,
        setPredictedLabel,
    ] = useState("");

    const [
        minScore,
        setMinScore,
    ] = useState(0);

    const [
        sharedOnly,
        setSharedOnly,
    ] = useState(false);

    const [
        baselineRegressionOnly,
        setBaselineRegressionOnly,
    ] = useState(false);

    const [
        offset,
        setOffset,
    ] = useState(0);

    const [
        items,
        setItems,
    ] = useState<ErrorItem[]>([]);

    const [
        total,
        setTotal,
    ] = useState(0);

    const [
        benchmarkNote,
        setBenchmarkNote,
    ] = useState("");

    const [
        confusions,
        setConfusions,
    ] = useState<ConfusionPair[]>([]);

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


    async function loadErrors(
        newOffset: number,
    ) {
        setLoading(true);
        setError(null);

        try {
            const response = await getErrors({
                trueLabel:
                    trueLabel || undefined,

                predictedLabel:
                    predictedLabel || undefined,

                minScore,
                maxScore: 1,

                sharedOnly,

                baselineRegressionOnly,

                offset: newOffset,
                limit: PAGE_SIZE,
            });

            setItems(response.items);
            setTotal(response.total);
            setOffset(response.offset);

            setBenchmarkNote(
                response.benchmark_note,
            );
        } catch (caught) {
            setItems([]);

            setError(
                caught instanceof Error
                    ? caught.message
                    : "Error Lab request failed",
            );
        } finally {
            setLoading(false);
        }
    }


    useEffect(() => {
        async function loadInitial() {
            try {
                const [
                    errorsResponse,
                    confusionResponse,
                ] = await Promise.all([
                    getErrors({
                        offset: 0,
                        limit: PAGE_SIZE,
                    }),

                    getConfusions(10),
                ]);

                setItems(
                    errorsResponse.items,
                );

                setTotal(
                    errorsResponse.total,
                );

                setBenchmarkNote(
                    errorsResponse
                        .benchmark_note,
                );

                setConfusions(
                    confusionResponse.items,
                );
            } catch (caught) {
                setError(
                    caught instanceof Error
                        ? caught.message
                        : "Unable to load Error Lab",
                );
            }
        }

        void loadInitial();
    }, []);


    async function handleSubmit(
        event: FormEvent,
    ) {
        event.preventDefault();

        await loadErrors(0);
    }


    async function resetFilters() {
        setTrueLabel("");
        setPredictedLabel("");
        setMinScore(0);
        setSharedOnly(false);
        setBaselineRegressionOnly(
            false,
        );

        setLoading(true);
        setError(null);

        try {
            const response = await getErrors({
                offset: 0,
                limit: PAGE_SIZE,
            });

            setItems(
                response.items,
            );

            setTotal(
                response.total,
            );

            setOffset(0);

            setBenchmarkNote(
                response.benchmark_note,
            );
        } catch (caught) {
            setError(
                caught instanceof Error
                    ? caught.message
                    : "Error Lab request failed",
            );
        } finally {
            setLoading(false);
        }
    }


    const pageStart =
        total === 0
            ? 0
            : offset + 1;

    const pageEnd = Math.min(
        offset + PAGE_SIZE,
        total,
    );

    const canPrevious =
        offset > 0;

    const canNext =
        offset + PAGE_SIZE < total;


    return (
        <div className="page">
            <section className="page-heading">
                <div className="eyebrow">
                    Error Lab
                </div>

                <h1>
                    Explore benchmark disagreements
                </h1>

                <p>
                    Inspect where TextCNN differs from
                    the dataset label, how those cases
                    cluster into confusion pairs, and
                    whether other model families agree.
                </p>
            </section>

            <section className="panel error-filter-panel">
                <form
                    className="error-filter-grid"
                    onSubmit={handleSubmit}
                >
                    <div>
                        <label
                            className="field-label"
                            htmlFor="true-label"
                        >
                            Benchmark label
                        </label>

                        <select
                            id="true-label"
                            className="select-field error-select"
                            value={trueLabel}
                            onChange={(event) =>
                                setTrueLabel(
                                    event.target.value,
                                )
                            }
                        >
                            {LABELS.map(
                                (label) => (
                                    <option
                                        value={label}
                                        key={
                                            label || "all"
                                        }
                                    >
                                        {label || "All"}
                                    </option>
                                ),
                            )}
                        </select>
                    </div>

                    <div>
                        <label
                            className="field-label"
                            htmlFor="predicted-label"
                        >
                            TextCNN prediction
                        </label>

                        <select
                            id="predicted-label"
                            className="select-field error-select"
                            value={predictedLabel}
                            onChange={(event) =>
                                setPredictedLabel(
                                    event.target.value,
                                )
                            }
                        >
                            {LABELS.map(
                                (label) => (
                                    <option
                                        value={label}
                                        key={
                                            label || "all"
                                        }
                                    >
                                        {label || "All"}
                                    </option>
                                ),
                            )}
                        </select>
                    </div>

                    <div>
                        <label
                            className="field-label"
                            htmlFor="minimum-score"
                        >
                            Minimum TextCNN score
                        </label>

                        <input
                            id="minimum-score"
                            className="range-field"
                            type="range"
                            min="0"
                            max="1"
                            step="0.05"
                            value={minScore}
                            onChange={(event) =>
                                setMinScore(
                                    Number(
                                        event.target.value,
                                    ),
                                )
                            }
                        />

                        <div className="range-value">
                            {(minScore * 100).toFixed(0)}
                            %
                        </div>
                    </div>

                    <div className="error-checkbox-group">
                        <label className="checkbox-row">
                            <input
                                type="checkbox"
                                checked={sharedOnly}
                                onChange={(event) =>
                                    setSharedOnly(
                                        event.target.checked,
                                    )
                                }
                            />

                            <span>
                                All-model shared
                                disagreements only
                            </span>
                        </label>

                        <label className="checkbox-row">
                            <input
                                type="checkbox"
                                checked={
                                    baselineRegressionOnly
                                }
                                onChange={(event) =>
                                    setBaselineRegressionOnly(
                                        event.target.checked,
                                    )
                                }
                            />

                            <span>
                                Baseline-correct /
                                TextCNN-wrong only
                            </span>
                        </label>
                    </div>

                    <div className="error-filter-actions">
                        <button
                            className="secondary-button error-reset-button"
                            type="button"
                            onClick={() => {
                                void resetFilters();
                            }}
                            disabled={loading}
                        >
                            Reset
                        </button>

                        <button
                            className="primary-button"
                            type="submit"
                            disabled={loading}
                        >
                            {loading
                                ? "Filtering..."
                                : "Apply filters"}
                        </button>
                    </div>
                </form>

                {error && (
                    <div className="alert alert-error">
                        {error}
                    </div>
                )}
            </section>

            <section className="error-overview-grid">
                <article className="panel error-stat-card">
                    <span className="metric-name">
                        Matching results
                    </span>

                    <strong>
                        {total}
                    </strong>
                </article>

                <article className="panel error-stat-card">
                    <span className="metric-name">
                        Default benchmark disagreements
                    </span>

                    <strong>
                        207
                    </strong>
                </article>

                <article className="panel error-stat-card">
                    <span className="metric-name">
                        Shared all-model disagreements
                    </span>

                    <strong>
                        93
                    </strong>
                </article>

                <article className="panel error-stat-card">
                    <span className="metric-name">
                        Baseline → TextCNN regressions
                    </span>

                    <strong>
                        76
                    </strong>
                </article>
            </section>

            <section className="panel error-caveat-panel">
                <div className="eyebrow">
                    Evaluation semantics
                </div>

                <p>
                    {benchmarkNote}
                </p>

                <p className="muted">
                    No systematic human relabeling or
                    annotation adjudication has been
                    performed.
                </p>
            </section>

            <section className="error-list-section">
                <div className="section-header">
                    <div>
                        <div className="eyebrow">
                            Cases
                        </div>

                        <h2>
                            Benchmark disagreements
                        </h2>
                    </div>

                    <span className="muted">
                        {pageStart}
                        –
                        {pageEnd}
                        {" "}
                        of
                        {" "}
                        {total}
                    </span>
                </div>

                {items.length === 0
                    ? (
                        <section className="panel">
                            <div className="empty-state">
                                <div className="empty-state-title">
                                    No matching cases
                                </div>

                                <p>
                                    Adjust the filters and try
                                    again.
                                </p>
                            </div>
                        </section>
                    )
                    : (
                        <div className="error-card-list">
                            {items.map(
                                (item) => (
                                    <ErrorCard
                                        key={item.example_id}
                                        item={item}
                                    />
                                ),
                            )}
                        </div>
                    )}

                <div className="pagination-row">
                    <button
                        className="secondary-button"
                        type="button"
                        disabled={
                            !canPrevious
                            || loading
                        }
                        onClick={() => {
                            void loadErrors(
                                Math.max(
                                    0,
                                    offset - PAGE_SIZE,
                                ),
                            );
                        }}
                    >
                        Previous
                    </button>

                    <button
                        className="secondary-button"
                        type="button"
                        disabled={
                            !canNext
                            || loading
                        }
                        onClick={() => {
                            void loadErrors(
                                offset + PAGE_SIZE,
                            );
                        }}
                    >
                        Next
                    </button>
                </div>
            </section>

            <section className="confusion-section">
                <div className="section-header">
                    <div>
                        <div className="eyebrow">
                            Confusions
                        </div>

                        <h2>
                            Dominant confusion pairs
                        </h2>
                    </div>
                </div>

                <section className="panel">
                    <ConfusionTable
                        items={confusions}
                    />

                    <div className="comparison-score-note">
                        TextCNN scores are raw softmax
                        probabilities and are not treated
                        as calibrated confidence values.
                    </div>
                </section>
            </section>
        </div>
    );
}