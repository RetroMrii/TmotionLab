# Current TmotionLab State

**Current phase:** P10_react_lab_frontend complete  
**Next phase:** P11_advanced_experiments

## Completed

### P0-P4 — foundation, data, baseline, and educational neural model

- Project renamed to TmotionLab.
- Core portfolio thesis locked as **Predict + Compare + Interpret + Error Lab**.
- GoEmotions / multi-label work deferred to advanced experiments.
- DAIR.AI Emotion single-label dataset pipeline implemented with stable example IDs.
- Dataset splits verified at 16,000 train / 2,000 validation / 2,000 test examples.
- Six labels: sadness, joy, love, anger, fear, surprise.
- Deterministic preprocessing and vocabulary artifacts implemented.
- TF-IDF + Logistic Regression baseline trained and frozen.
- Manual NumPy MLP implemented, trained, and promoted as the educational neural reference.

### P5 — TextCNN model

- PyTorch TextCNN implemented with embedding, Conv1d kernels, ReLU, global max pooling, dropout, and linear classification.
- Promoted model: `textcnn_dair_v001`.
- Test accuracy: **0.8965**.
- Test macro F1: **0.850952...** (reported as 0.8510).
- Test weighted F1: **0.8967**.

### P6 — model comparison lab

- Baseline, NumPy MLP, and TextCNN predictions aligned on the same 2,000 held-out examples using stable IDs.
- Reproducible comparison CSV/JSON artifacts generated.
- Key comparison findings:
  - all-model agreement: 0.8265
  - baseline wrong -> TextCNN correct: 205
  - baseline correct -> TextCNN wrong: 76
  - NumPy wrong -> TextCNN correct: 115
  - all models agree and disagree with benchmark: 93
  - three-way disagreements: 21
- Generated model comparison report completed.

### P7 — TextCNN interpretation lab

- Architecture-native TextCNN activation tracing implemented.
- Strong convolutional activations can be mapped back to exact token spans.
- Padding-overlap maxima are tracked explicitly instead of silently discarded.
- Interpretation artifacts and aggregate reports generated.
- Interpretation wording locked: observed activation is model-internal evidence, not causal class attribution or psychological ground truth.

### P8 — Error Lab

- Reproducible benchmark-relative Error Lab implemented.
- TextCNN has **207** test-set benchmark disagreements.
- Shared all-model disagreements: **93**.
- Baseline-correct / TextCNN-wrong regressions: **76**.
- Confusion-pair and high-score disagreement artifacts generated.
- Manual relabeling / systematic human adjudication intentionally deferred.
- User-facing language uses **benchmark disagreement** / **evaluation error** rather than assuming every mismatch is a semantic model failure.

### P9 — FastAPI backend

- FastAPI backend implemented and frozen at API version **0.3.0** for the first frontend implementation.
- TextCNN is loaded once through FastAPI lifespan startup.
- Analysis artifacts are loaded once through a shared artifact store.
- Implemented endpoints:
  - `GET /health`
  - `GET /models`
  - `POST /predict`
  - `GET /metrics`
  - `GET /compare/{example_id}`
  - `POST /interpret`
  - `GET /errors`
  - `GET /errors/confusions`
- Consistent API validation/error responses implemented.
- CORS configured for the React/Vite development origin.
- Backend smoke-test CLI implemented.

### P10 — React lab frontend

- React + TypeScript + Vite frontend implemented.
- Shared typed API client added.
- Implemented views:
  - Home / model registry
  - Predict
  - Compare
  - Interpret
  - Error Lab
  - Metrics
  - About / Methodology
- Added frontend 404 handling and responsive layout behavior.
- Predict defaults to top-3 probabilities with optional full six-class distribution.
- Compare remains held-out-example based and uses stable test example IDs.
- Interpret displays exact token regions, kernel/filter metadata, activation magnitudes, and padding diagnostics.
- Error Lab supports label filters, score thresholding, shared-disagreement filtering, TextCNN-regression filtering, pagination, and confusion-pair summaries.
- Metrics view compares accuracy, macro F1, and weighted F1 across the three promoted/reference models.
- About page documents methodology, limitations, benchmark semantics, and non-diagnostic intended use.

## Promoted / reference models

| Model | Role | Test accuracy | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|
| `baseline_tfidf_dair_v001` | baseline | 0.8320 | 0.7301 | 0.8208 |
| `numpy_mlp_dair_v002` | educational neural reference | 0.8770 | 0.8058 | 0.8742 |
| `textcnn_dair_v001` | promoted production model | 0.8965 | 0.8510 | 0.8967 |

## Current architecture

```text
DAIR.AI Emotion
      |
      v
preprocessing / stable IDs
      |
      +-------------------+-------------------+
      |                   |                   |
      v                   v                   v
TF-IDF + LR          NumPy MLP            TextCNN
      |                   |                   |
      +-------------------+-------------------+
                          |
                          v
                saved evaluation artifacts
                          |
               +----------+----------+
               |                     |
               v                     v
         comparison/error       live TextCNN
             artifacts          predict/interpret
               |                     |
               +----------+----------+
                          v
                       FastAPI
                          |
                          v
                    React / Vite UI
```

## Current decisions

See `docs/decision_log.md`.

Important locked decisions:

- Core thesis remains **Predict + Compare + Interpret + Error Lab**.
- GoEmotions and multi-label modeling remain deferred to P11/stretch work.
- Comparison must use stable held-out example IDs and the exact same examples across models.
- Raw scores from different model families are not assumed to be calibrated or directly comparable.
- Softmax probability is not described as calibrated confidence.
- TextCNN activation spans are architecture-native evidence, not causal explanations.
- Error Lab remains benchmark-relative; no systematic human adjudication has been performed.
- Only TextCNN is loaded for live inference in the current backend. Baseline and NumPy results are served from saved benchmark artifacts.

## Current errors or risks

- No known blocking functional issue remains from P0-P10 if the full backend/frontend validation gates are green locally.
- The Error Lab still contains dataset-label ambiguity that has not been manually adjudicated; this is documented rather than hidden.
- Model probabilities have not been calibrated.
- Frontend production deployment configuration has not yet been finalized.
- Advanced experiments must not destabilize the completed P0-P10 core.

## Next exact step

Before beginning P11, checkpoint the completed backend/frontend milestone in Git and keep the P9 API contract stable.

Then enter `P11_advanced_experiments` only with a controlled experiment that adds clear portfolio value. Candidate work should be evaluated against reproducibility, interpretability, and whether it changes the product story enough to justify additional complexity.
