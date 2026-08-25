import type {
    EmotionProbability,
} from "../types/api";

interface ProbabilityBarProps {
    item: EmotionProbability;
}

export function ProbabilityBar({
    item,
}: ProbabilityBarProps) {
    const percentage =
        item.probability * 100;

    return (
        <div className="probability-row">
            <div className="probability-label-row">
                <span className="emotion-label">
                    {item.label}
                </span>

                <span className="probability-value">
                    {percentage.toFixed(2)}%
                </span>
            </div>

            <div className="probability-track">
                <div
                    className="probability-fill"
                    style={{
                        width: `${percentage}%`,
                    }}
                />
            </div>
        </div>
    );
}