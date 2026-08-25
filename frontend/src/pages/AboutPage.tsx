const LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
];

const PHASES = [
    {
        title: "Predict",
        text:
            "Run the promoted TextCNN on new English text "
            + "and inspect its top softmax probabilities.",
    },
    {
        title: "Compare",
        text:
            "Compare the three saved model predictions on "
            + "the exact same held-out benchmark example.",
    },
    {
        title: "Interpret",
        text:
            "Map strong TextCNN convolutional activations "
            + "back to local token spans.",
    },
    {
        title: "Error Lab",
        text:
            "Explore benchmark disagreements, confusion "
            + "pairs, shared errors, and regressions.",
    },
];


export function AboutPage() {
    return (
        <div className="page">
            <section className="page-heading">
                <div className="eyebrow">
                    About
                </div>

                <h1>
                    TmotionLab methodology
                </h1>

                <p>
                    TmotionLab is an interactive NLP
                    laboratory for studying how several
                    text-classification model families
                    behave on the same emotion-classification
                    problem.
                </p>
            </section>

            <section className="about-thesis-grid">
                {PHASES.map(
                    (item) => (
                        <article
                            className="panel about-thesis-card"
                            key={item.title}
                        >
                            <div className="eyebrow">
                                {item.title}
                            </div>

                            <p>
                                {item.text}
                            </p>
                        </article>
                    ),
                )}
            </section>

            <section className="panel about-section">
                <div className="eyebrow">
                    Dataset
                </div>

                <h2>
                    Six-class emotion benchmark
                </h2>

                <p>
                    The project uses the
                    {" "}
                    <strong>
                        dair-ai/emotion
                    </strong>
                    {" "}
                    dataset configuration used throughout
                    the training pipeline.
                </p>

                <div className="about-stat-grid">
                    <div>
                        <span className="metric-name">
                            Training examples
                        </span>

                        <strong>
                            16,000
                        </strong>
                    </div>

                    <div>
                        <span className="metric-name">
                            Validation examples
                        </span>

                        <strong>
                            2,000
                        </strong>
                    </div>

                    <div>
                        <span className="metric-name">
                            Test examples
                        </span>

                        <strong>
                            2,000
                        </strong>
                    </div>

                    <div>
                        <span className="metric-name">
                            Classes
                        </span>

                        <strong>
                            6
                        </strong>
                    </div>
                </div>

                <div className="label-chip-row">
                    {LABELS.map(
                        (label) => (
                            <span
                                className="label-chip"
                                key={label}
                            >
                                {label}
                            </span>
                        ),
                    )}
                </div>
            </section>

            <section className="about-model-grid">
                <article className="panel about-model-card">
                    <div className="eyebrow">
                        Baseline
                    </div>

                    <h2>
                        TF-IDF + Logistic Regression
                    </h2>

                    <p>
                        A classical sparse-text baseline used
                        as the reference point for later model
                        comparisons.
                    </p>

                    <div className="about-model-metrics">
                        <span>
                            Accuracy
                            {" "}
                            <strong>
                                83.20%
                            </strong>
                        </span>

                        <span>
                            Macro F1
                            {" "}
                            <strong>
                                0.7301
                            </strong>
                        </span>
                    </div>
                </article>

                <article className="panel about-model-card">
                    <div className="eyebrow">
                        Educational neural model
                    </div>

                    <h2>
                        NumPy MLP
                    </h2>

                    <p>
                        A manually implemented feed-forward
                        neural network using NumPy, including
                        explicit forward propagation,
                        backpropagation, and optimization.
                    </p>

                    <div className="about-model-metrics">
                        <span>
                            Accuracy
                            {" "}
                            <strong>
                                87.70%
                            </strong>
                        </span>

                        <span>
                            Macro F1
                            {" "}
                            <strong>
                                0.8058
                            </strong>
                        </span>
                    </div>
                </article>

                <article className="panel about-model-card about-model-promoted">
                    <div className="eyebrow">
                        Promoted model
                    </div>

                    <h2>
                        PyTorch TextCNN
                    </h2>

                    <p>
                        An embedding-based convolutional text
                        classifier using multiple kernel sizes,
                        global max pooling, dropout, and a
                        linear classification head.
                    </p>

                    <div className="about-model-metrics">
                        <span>
                            Accuracy
                            {" "}
                            <strong>
                                89.65%
                            </strong>
                        </span>

                        <span>
                            Macro F1
                            {" "}
                            <strong>
                                0.8510
                            </strong>
                        </span>
                    </div>
                </article>
            </section>

            <section className="panel about-section">
                <div className="eyebrow">
                    Interpretation
                </div>

                <h2>
                    Architecture-native activation tracing
                </h2>

                <p>
                    TextCNN interpretation is based on the
                    strongest post-ReLU convolutional
                    activations produced before global max
                    pooling. The activation position can be
                    mapped back to the corresponding input
                    n-gram.
                </p>

                <div className="about-do-dont-grid">
                    <div>
                        <h3>
                            What it supports
                        </h3>

                        <p>
                            Identifying local token regions that
                            strongly activate individual TextCNN
                            filters.
                        </p>
                    </div>

                    <div>
                        <h3>
                            What it does not support
                        </h3>

                        <p>
                            Claiming that a highlighted phrase
                            caused the final class prediction,
                            measuring calibrated class
                            importance, or identifying a
                            person&apos;s actual psychological
                            state.
                        </p>
                    </div>
                </div>
            </section>

            <section className="panel about-section">
                <div className="eyebrow">
                    Error Lab semantics
                </div>

                <h2>
                    Benchmark disagreement is not always
                    semantic failure
                </h2>

                <p>
                    TmotionLab treats a prediction that
                    differs from the dataset label as an
                    evaluation error for reproducible
                    benchmark analysis.
                </p>

                <p>
                    Some cases may nevertheless contain
                    ambiguous emotion boundaries, mixed
                    emotions, limited textual context, or
                    questionable annotations. Agreement
                    among several models is evidence worth
                    inspecting, but it does not prove that
                    the benchmark label is incorrect.
                </p>

                <div className="alert alert-info">
                    No systematic human relabeling,
                    annotation adjudication, or
                    multi-reviewer agreement study was
                    performed.
                </div>
            </section>

            <section className="panel about-section">
                <div className="eyebrow">
                    Scores
                </div>

                <h2>
                    Probability and calibration limits
                </h2>

                <p>
                    TextCNN uses softmax probabilities for
                    ranking predictions. These values have
                    not been calibrated and should not be
                    interpreted as externally validated
                    confidence estimates.
                </p>

                <p>
                    Raw scores from different model
                    families are also not assumed to be
                    directly comparable because each model
                    produces scores using a different
                    representation and decision function.
                </p>
            </section>

            <section className="panel about-section">
                <div className="eyebrow">
                    System architecture
                </div>

                <h2>
                    Training artifacts separated from serving
                </h2>

                <div className="architecture-flow">
                    <div>
                        Dataset
                    </div>

                    <span>
                        →
                    </span>

                    <div>
                        Training pipeline
                    </div>

                    <span>
                        →
                    </span>

                    <div>
                        Saved models &amp; analysis artifacts
                    </div>

                    <span>
                        →
                    </span>

                    <div>
                        FastAPI
                    </div>

                    <span>
                        →
                    </span>

                    <div>
                        React laboratory
                    </div>
                </div>

                <p className="comparison-score-note">
                    Offline benchmark analysis remains
                    reproducible through saved artifacts,
                    while live inference and interpretation
                    are served through the promoted TextCNN.
                </p>
            </section>

            <section className="panel about-section">
                <div className="eyebrow">
                    Intended use
                </div>

                <h2>
                    Educational and analytical
                </h2>

                <p>
                    TmotionLab is designed to demonstrate
                    machine-learning engineering,
                    comparative evaluation, model behavior,
                    and neural-network interpretation.
                </p>

                <div className="about-limit-banner">
                    It is not a mental-health diagnostic
                    system and should not be used to infer
                    clinical or psychological conditions.
                </div>
            </section>
        </div>
    );
}