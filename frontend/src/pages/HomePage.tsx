import {
    useEffect,
    useState,
} from "react";

import {
    getHealth,
    getModels,
} from "../api/client";

import {
    StatusBadge,
} from "../components/StatusBadge";

import type {
    HealthResponse,
    ModelMetadata,
} from "../types/api";


export function HomePage() {
    const [
        health,
        setHealth,
    ] = useState<HealthResponse | null>(
        null,
    );

    const [
        models,
        setModels,
    ] = useState<ModelMetadata[]>([]);

    const [
        error,
        setError,
    ] = useState<string | null>(
        null,
    );

    useEffect(() => {
        async function load() {
            try {
                const [
                    healthResponse,
                    modelsResponse,
                ] = await Promise.all([
                    getHealth(),
                    getModels(),
                ]);

                setHealth(
                    healthResponse,
                );

                setModels(
                    modelsResponse.models,
                );
            } catch (caught) {
                setError(
                    caught instanceof Error
                        ? caught.message
                        : "Unable to connect to API",
                );
            }
        }

        void load();
    }, []);

    return (
        <div className="page">
            <section className="hero">
                <div>
                    <div className="eyebrow">
                        Interactive NLP laboratory
                    </div>

                    <h1>
                        Inspect how emotion classifiers
                        behave—not just what they predict.
                    </h1>

                    <p className="hero-copy">
                        TmotionLab compares three model
                        families, exposes TextCNN activation
                        regions, and turns benchmark
                        disagreements into an inspectable
                        Error Lab.
                    </p>
                </div>

                <div>
                    <StatusBadge
                        online={
                            health?.status === "ok"
                        }
                    />
                </div>
            </section>

            {error && (
                <div className="alert alert-error">
                    Backend connection failed:
                    {" "}
                    {error}
                </div>
            )}

            <section className="section">
                <div className="section-header">
                    <div>
                        <div className="eyebrow">
                            Model registry
                        </div>

                        <h2>
                            Promoted models
                        </h2>
                    </div>

                    {health && (
                        <span className="muted">
                            API {health.version}
                        </span>
                    )}
                </div>

                <div className="model-grid">
                    {models.map(
                        (model) => (
                            <article
                                className="model-card"
                                key={model.model_id}
                            >
                                <div className="model-card-top">
                                    <span className="model-role">
                                        {model.role}
                                    </span>

                                    {model.runtime_loaded && (
                                        <span className="live-badge">
                                            live
                                        </span>
                                    )}
                                </div>

                                <h3>
                                    {model.display_name}
                                </h3>

                                <div className="muted">
                                    {model.implementation}
                                </div>

                                <div className="metric-row">
                                    <div>
                                        <span className="metric-name">
                                            Accuracy
                                        </span>

                                        <strong>
                                            {(
                                                model.accuracy * 100
                                            ).toFixed(2)}
                                            %
                                        </strong>
                                    </div>

                                    <div>
                                        <span className="metric-name">
                                            Macro F1
                                        </span>

                                        <strong>
                                            {model.macro_f1.toFixed(
                                                4,
                                            )}
                                        </strong>
                                    </div>
                                </div>
                            </article>
                        ),
                    )}
                </div>
            </section>
        </div>
    );
}