import type {
    ErrorItem,
} from "../types/api";


interface ErrorCardProps {
    item: ErrorItem;
}


export function ErrorCard({
    item,
}: ErrorCardProps) {
    return (
        <article className="error-card">
            <div className="error-card-header">
                <div>
                    <div className="eyebrow">
                        #{item.error_rank}
                    </div>

                    <h3>
                        {item.example_id}
                    </h3>
                </div>

                <div className="error-badge-row">
                    {item.high_score_error && (
                        <span className="error-badge error-badge-score">
                            high score
                        </span>
                    )}

                    {item.all_agree_and_wrong && (
                        <span className="error-badge error-badge-shared">
                            all agree
                        </span>
                    )}

                    {item.baseline_correct_textcnn_wrong && (
                        <span className="error-badge error-badge-regression">
                            TextCNN regression
                        </span>
                    )}

                    {item.three_way_disagreement && (
                        <span className="error-badge">
                            three-way
                        </span>
                    )}
                </div>
            </div>

            <p className="error-text">
                {item.text}
            </p>

            <div className="error-label-grid">
                <div>
                    <span className="metric-name">
                        Benchmark
                    </span>

                    <strong className="emotion-value">
                        {item.true_label_name}
                    </strong>
                </div>

                <div>
                    <span className="metric-name">
                        TextCNN
                    </span>

                    <strong className="emotion-value">
                        {item.textcnn_prediction}
                    </strong>

                    <span className="error-score">
                        {(item.textcnn_score * 100).toFixed(2)}
                        %
                    </span>
                </div>

                <div>
                    <span className="metric-name">
                        Baseline
                    </span>

                    <strong className="emotion-value">
                        {item.baseline_prediction}
                    </strong>
                </div>

                <div>
                    <span className="metric-name">
                        NumPy MLP
                    </span>

                    <strong className="emotion-value">
                        {item.numpy_prediction}
                    </strong>
                </div>
            </div>

            <div className="error-card-footer">
                <span>
                    {item.confusion_pair}
                </span>

                {item.interpretation_available && (
                    <span>
                        interpretation available
                    </span>
                )}
            </div>
        </article>
    );
}