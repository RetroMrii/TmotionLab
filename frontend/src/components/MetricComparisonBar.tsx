interface MetricComparisonBarProps {
    label: string;
    value: number;
    maxValue?: number;
}

export function MetricComparisonBar({
    label,
    value,
    maxValue = 1,
}: MetricComparisonBarProps) {
    const percentage =
        Math.min(
            100,
            Math.max(
                0,
                (value / maxValue) * 100,
            ),
        );

    return (
        <div className="metric-comparison-row">
            <div className="metric-comparison-header">
                <span>
                    {label}
                </span>

                <strong>
                    {(value * 100).toFixed(2)}%
                </strong>
            </div>

            <div className="metric-comparison-track">
                <div
                    className="metric-comparison-fill"
                    style={{
                        width: `${percentage}%`,
                    }}
                />
            </div>
        </div>
    );
}