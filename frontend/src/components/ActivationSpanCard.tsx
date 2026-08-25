import type {
    ActivationSpan,
} from "../types/api";

interface ActivationSpanCardProps {
    rank: number;
    span: ActivationSpan;
}

export function ActivationSpanCard({
    rank,
    span,
}: ActivationSpanCardProps) {
    return (
        <article className="activation-card">
            <div className="activation-card-header">
                <div className="activation-rank">
                    #{rank}
                </div>

                <div className="activation-value">
                    {span.activation.toFixed(4)}
                </div>
            </div>

            <div className="activation-span-text">
                “{span.span_text}”
            </div>

            <div className="activation-meta-grid">
                <div>
                    <span className="metric-name">
                        Kernel
                    </span>

                    <strong>
                        {span.kernel_size}
                    </strong>
                </div>

                <div>
                    <span className="metric-name">
                        Filter
                    </span>

                    <strong>
                        {span.filter_index}
                    </strong>
                </div>

                <div>
                    <span className="metric-name">
                        Tokens
                    </span>

                    <strong>
                        {span.token_start}
                        –
                        {span.token_end_exclusive - 1}
                    </strong>
                </div>
            </div>
        </article>
    );
}