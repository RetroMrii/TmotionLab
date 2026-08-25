interface ModelPredictionCardProps {
    name: string;
    role: string;
    prediction: string;
    score: number;
    correct: boolean;
}

export function ModelPredictionCard({
    name,
    role,
    prediction,
    score,
    correct,
}: ModelPredictionCardProps) {
    return (
        <article
            className={
                correct
                    ? "comparison-model-card comparison-model-correct"
                    : "comparison-model-card comparison-model-wrong"
            }
        >
            <div className="comparison-model-header">
                <div>
                    <div className="model-role">
                        {role}
                    </div>

                    <h3>
                        {name}
                    </h3>
                </div>

                <span
                    className={
                        correct
                            ? "comparison-status comparison-status-correct"
                            : "comparison-status comparison-status-wrong"
                    }
                >
                    {correct
                        ? "benchmark match"
                        : "benchmark disagreement"}
                </span>
            </div>

            <div className="comparison-prediction">
                {prediction}
            </div>

            <div className="comparison-score">
                {(score * 100).toFixed(2)}%
            </div>

            <div className="muted">
                Raw model score
            </div>
        </article>
    );
}