# TmotionLab

TmotionLab is an interactive NLP laboratory for studying how emotion-classification models behave on English text. The project goes beyond a single prediction demo: it compares multiple modeling approaches, exposes model confidence, inspects TextCNN activation patterns, and turns model failures into a first-class Error Lab.

> **Status:** planning / foundation. The repository is being built incrementally with reproducible commands, tests, experiment logs, and validation gates.

## Project Thesis

Most student emotion-classification projects stop at `text -> label`. TmotionLab is designed around a stronger question:

> How do different text-classification models perceive emotional language, where do they agree or disagree, what textual regions strongly activate the neural model, and what systematic failure modes remain?

The project therefore treats prediction as only one layer of the system.

## Core Experience

TmotionLab will provide four connected laboratory views:

1. **Predict** — classify text and show the top three emotion probabilities.
2. **Compare** — compare TF-IDF + Logistic Regression, a NumPy neural network, and a PyTorch TextCNN on the same examples and evaluation sets.
3. **Interpret** — inspect high-activation n-grams / convolutional regions from TextCNN as model-behavior evidence. These signals are presented as interpretation aids, not as complete causal explanations.
4. **Error Lab** — explore confusion pairs, high-confidence mistakes, ambiguity, negation failures, semantic overlap, and other recurring error categories.

## Model Stack

| Role | Model | Purpose |
|---|---|---|
| Baseline | TF-IDF + Logistic Regression | Establish a strong, fast classical reference |
| Learning model | NumPy neural network | Demonstrate neural-network fundamentals from scratch |
| Main model | PyTorch TextCNN | Production inference and convolutional text interpretation |

The first stable dataset target is a verified single-label emotion dataset such as DAIR.AI Emotion. GoEmotions and multi-label modeling are advanced extensions, not prerequisites for the core portfolio story.

## Application Stack

```text
ML:       Python + NumPy + scikit-learn + PyTorch
Backend:  FastAPI
Frontend: React
Tracking: JSON experiment records + CSV epoch logs
Testing:  pytest + frontend lint/build checks
```

## Engineering Principles

- Reproducible commands instead of hidden notebook state.
- Small smoke tests before expensive training.
- Dataset schemas are verified, never assumed.
- Every promoted model has metrics, config snapshots, artifacts, and error analysis.
- Model comparison uses the same held-out split and compatible preprocessing/evaluation contracts.
- Interpretation output is validated and never presented as psychological ground truth.
- Emotion labels are dataset-defined categories, not clinical diagnoses or direct measurements of a person's internal state.

## Planned Repository Areas

```text
TmotionLab/
├── ml/                 # datasets, models, training, evaluation, interpretation
├── backend/            # FastAPI service
├── frontend/           # React laboratory interface
├── tests/              # ML and integration tests
├── reports/            # comparison, interpretation, and error-analysis reports
├── docs/               # contracts, dataset notes, model cards, decisions
└── PROJECT_PLAN.md     # detailed internal specification
```

## Current Development Order

```text
foundation
-> dataset pipeline
-> classical baseline
-> NumPy neural model
-> TextCNN
-> comparison lab
-> interpretation lab
-> error lab
-> FastAPI
-> React laboratory UI
-> advanced datasets / multi-label experiments
-> portfolio polish
```

## Documentation

- `PROJECT_PLAN.md` — detailed product and engineering plan.
- `MAIN_PROMPT.md` — operating contract for AI-assisted development.
- `CONTEXT_ENGINEERING.md` — source-of-truth and project-state rules.
- `HARNESS_ENGINEERING.md` — reproducibility, tests, artifact contracts, and promotion gates.
- `docs/decision_log.md` — accepted, changed, and deprecated project decisions.

## Current Limitations

TmotionLab is an educational and portfolio project. Its predictions are probabilistic, depend on dataset definitions, and can fail on sarcasm, ambiguity, mixed emotions, cultural context, and short or underspecified text. It must not be presented as a mental-health diagnostic system.
