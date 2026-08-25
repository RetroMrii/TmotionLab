import {
    useEffect,
    useMemo,
    useState,
} from "react";

import {
    getMetrics,
} from "../api/client";

import {
    MetricComparisonBar,
} from "../components/MetricComparisonBar";

import type {
    ModelMetrics,
} from "../types/api";


const MODEL_ORDER = [
    "baseline_tfidf_dair_v001",
    "numpy_mlp_dair_v002",
    "textcnn_dair_v001",
];


const MODEL_NAMES: Record<
    string,
    string
> = {
    baseline_tfidf_dair_v001:
        "TF-IDF + Logistic Regression",

    numpy_mlp_dair_v002:
        "NumPy MLP",

    textcnn_dair_v001:
        "TextCNN",
};


const MODEL_ROLES: Record<
    string,
    string
> = {
    baseline_tfidf_dair_v001:
        "Baseline",

    numpy_mlp_dair_v002:
        "Educational neural model",

    textcnn_dair_v001:
        "Promoted production model",
};


function sortModels(
    models: ModelMetrics[],
): ModelMetrics[] {
    return [...models].sort(
        (a, b) =>
            MODEL_ORDER.indexOf(
                a.model_id,
            )
            - MODEL_ORDER.indexOf(
                b.model_id,
            ),
    );
}


function percentagePointChange(
    current: number,
    previous: number,
): number {
    return (
        current - previous
    ) * 100;
}


export function MetricsPage() {
    const [
        models,
        setModels,
    ] = useState<ModelMetrics[]>([]);

    const [
        loading,
        setLoading,
    ] = useState(true);

    const [
        error,
        setError,
    ] = useState<string | null>(
        null,
    );


    useEffect(() => {
        async function load() {
            try {
                const response =
                    await getMetrics();

                setModels(
                    sortModels(
                        response.models,
                    ),
                );
            } catch (caught) {
                setError(
                    caught instanceof Error
                        ? caught.message
                        : "Unable to load metrics",
                );
            } finally {
                setLoading(false);
            }
        }

        void load();
    }, []);


    const summary = useMemo(() => {
        if (models.length !== 3) {
            return null;
        }

        const [
            baseline,
            numpy,
            textcnn,
        ] = models;

        return {
            baselineToNumpyAccuracy:
                percentagePointChange(
                    numpy.accuracy,
                    baseline.accuracy,
                ),

            numpyToTextcnnAccuracy:
                percentagePointChange(
                    textcnn.accuracy,
                    numpy.accuracy,
                ),

            baselineToTextcnnAccuracy:
                percentagePointChange(
                    textcnn.accuracy,
                    baseline.accuracy,
                ),

            baselineToTextcnnMacroF1:
                percentagePointChange(
                    textcnn.macro_f1,
                    baseline.macro_f1,
                ),
        };
    }, [models]);


    return (
        <div className="page">
            <section className="page-heading">
                <div className="eyebrow">
                    Metrics
                </div>

                <h1>
                    Benchmark performance across three model families
                </h1>

                <p>
                    Compare the promoted models using the
                    same held-out test set and the saved
                    evaluation metrics generated during
                    training and analysis.
                </p>
            </section>

            {loading && (
                <section className="panel">
                    <div className="empty-state">
                        <div className="empty-state-title">
                            Loading metrics
                        </div>
                    </div>
                </section>
            )}

            {error && (
                <div className="alert alert-error">
                    {error}
                </div>
            )}

            {!loading
                && !error
                && (
                    <>
                        <section className="metrics-model-grid">
                            {models.map(
                                (model) => (
                                    <article
                                        className={
                                            model.model_id
                                                === "textcnn_dair_v001"
                                                ? "panel metrics-model-card metrics-model-card-promoted"
                                                : "panel metrics-model-card"
                                        }
                                        key={model.model_id}
                                    >
                                        <div className="metrics-card-header">
                                            <div>
                                                <div className="eyebrow">
                                                    {
                                                        MODEL_ROLES[
                                                        model.model_id
                                                        ]
                                                    }
                                                </div>

                                                <h2>
                                                    {
                                                        MODEL_NAMES[
                                                        model.model_id
                                                        ]
                                                    }
                                                </h2>
                                            </div>

                                            {model.model_id
                                                === "textcnn_dair_v001"
                                                && (
                                                    <span className="live-badge">
                                                        promoted
                                                    </span>
                                                )}
                                        </div>

                                        <div className="metrics-card-values">
                                            <div>
                                                <span className="metric-name">
                                                    Accuracy
                                                </span>

                                                <strong>
                                                    {(
                                                        model.accuracy
                                                        * 100
                                                    ).toFixed(2)}
                                                    %
                                                </strong>
                                            </div>

                                            <div>
                                                <span className="metric-name">
                                                    Macro F1
                                                </span>

                                                <strong>
                                                    {
                                                        model.macro_f1.toFixed(
                                                            4,
                                                        )
                                                    }
                                                </strong>
                                            </div>

                                            <div>
                                                <span className="metric-name">
                                                    Weighted F1
                                                </span>

                                                <strong>
                                                    {
                                                        model.weighted_f1.toFixed(
                                                            4,
                                                        )
                                                    }
                                                </strong>
                                            </div>
                                        </div>

                                        <div className="metrics-bars">
                                            <MetricComparisonBar
                                                label="Accuracy"
                                                value={
                                                    model.accuracy
                                                }
                                            />

                                            <MetricComparisonBar
                                                label="Macro F1"
                                                value={
                                                    model.macro_f1
                                                }
                                            />

                                            <MetricComparisonBar
                                                label="Weighted F1"
                                                value={
                                                    model.weighted_f1
                                                }
                                            />
                                        </div>

                                        <div className="model-note">
                                            {model.model_id}
                                        </div>
                                    </article>
                                ),
                            )}
                        </section>

                        {summary && (
                            <section className="panel metrics-progress-panel">
                                <div className="section-header">
                                    <div>
                                        <div className="eyebrow">
                                            Progression
                                        </div>

                                        <h2>
                                            Model improvement
                                        </h2>
                                    </div>
                                </div>

                                <div className="metrics-improvement-grid">
                                    <div>
                                        <span className="metric-name">
                                            Baseline → NumPy MLP
                                        </span>

                                        <strong>
                                            +
                                            {
                                                summary
                                                    .baselineToNumpyAccuracy
                                                    .toFixed(2)
                                            }
                                            {" "}
                                            pp
                                        </strong>

                                        <span className="muted">
                                            accuracy
                                        </span>
                                    </div>

                                    <div>
                                        <span className="metric-name">
                                            NumPy MLP → TextCNN
                                        </span>

                                        <strong>
                                            +
                                            {
                                                summary
                                                    .numpyToTextcnnAccuracy
                                                    .toFixed(2)
                                            }
                                            {" "}
                                            pp
                                        </strong>

                                        <span className="muted">
                                            accuracy
                                        </span>
                                    </div>

                                    <div>
                                        <span className="metric-name">
                                            Baseline → TextCNN
                                        </span>

                                        <strong>
                                            +
                                            {
                                                summary
                                                    .baselineToTextcnnAccuracy
                                                    .toFixed(2)
                                            }
                                            {" "}
                                            pp
                                        </strong>

                                        <span className="muted">
                                            accuracy
                                        </span>
                                    </div>

                                    <div>
                                        <span className="metric-name">
                                            Baseline → TextCNN
                                        </span>

                                        <strong>
                                            +
                                            {
                                                summary
                                                    .baselineToTextcnnMacroF1
                                                    .toFixed(2)
                                            }
                                            {" "}
                                            pp
                                        </strong>

                                        <span className="muted">
                                            macro F1
                                        </span>
                                    </div>
                                </div>

                                <div className="alert alert-info">
                                    Percentage-point changes summarize
                                    differences on this fixed benchmark.
                                    No statistical significance test is
                                    implied.
                                </div>
                            </section>
                        )}

                        <section className="panel metrics-method-panel">
                            <div className="eyebrow">
                                Evaluation context
                            </div>

                            <h2>
                                Same test split, different model families
                            </h2>

                            <div className="metrics-method-grid">
                                <div>
                                    <h3>
                                        Accuracy
                                    </h3>

                                    <p>
                                        Fraction of held-out examples
                                        whose predicted label matches
                                        the dataset label.
                                    </p>
                                </div>

                                <div>
                                    <h3>
                                        Macro F1
                                    </h3>

                                    <p>
                                        Average F1 across the six
                                        emotion classes with equal
                                        weight per class.
                                    </p>
                                </div>

                                <div>
                                    <h3>
                                        Weighted F1
                                    </h3>

                                    <p>
                                        F1 averaged across classes
                                        with weighting based on class
                                        support.
                                    </p>
                                </div>
                            </div>

                            <div className="comparison-score-note">
                                Metrics are benchmark-relative.
                                They measure agreement with the
                                dataset labels and do not establish
                                psychological truth or semantic
                                correctness for every individual
                                example.
                            </div>
                        </section>
                    </>
                )}
        </div>
    );
}