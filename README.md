# TmotionLab

**An evolving NLP engineering laboratory for fine-grained, multi-label emotion classification and data-centric experimentation.**

TmotionLab began as a complete six-class DAIR.AI emotion-classification laboratory with **Predict, Compare, Interpret, and Error Lab** views. The project is now transitioning to the 28-label **GoEmotions** benchmark, with a new independent ML pipeline, a multi-label TF-IDF baseline, and a fine-tuned RoBERTa classifier. Planned research will examine annotation disagreement and whether quality-controlled synthetic training examples improve performance beyond real-data resampling.

> **Project status (October 2026):** P11 development in progress. GoEmotions data preparation, the TF-IDF baseline, and the first RoBERTa training run are complete. Synthetic-data experiments and integration of the new models into the website are **not complete**. The legacy FastAPI/React app was built for DAIR and may not run against the restructured ML package until integration work is performed.

## Current GoEmotions research

### Data

- Source: [`google-research-datasets/go_emotions`](https://huggingface.co/datasets/google-research-datasets/go_emotions), `simplified` and `raw` configurations.
- Pinned dataset revision: `add492243ff905527e67aeb8b80c082af02207c3`.
- Task: English multi-label classification with **27 emotions plus neutral** (28 labels).
- Official `simplified` splits: 43,410 train / 5,426 validation / 5,427 test.
- Training view `goemotions_v001`: **43,252** training examples after excluding **158** rows whose normalized text overlaps validation or test. Official validation and test membership is preserved.
- Remaining evaluation caveat: **11 normalized texts overlap validation and test**. Exact/normalized-text checks do not rule out near-duplicates.
- Raw annotation snapshot: **211,225 rater-annotation rows** spanning **58,011 unique comment IDs**. This is *not* 211,225 independent examples.

Source snapshots are stored separately from processed training data. Dataset manifests record source revision, row counts, label mapping, and SHA-256 checksums. Raw annotations are reserved for annotation/disagreement research; they are not automatically merged into model training.

## Research roadmap

| Stage | State | Focus |
|---|---|---|
| P11-B | Complete | Revision-pinned GoEmotions ingestion, raw-annotation audit, leakage-controlled training view |
| P11-C | Complete | Multi-label TF-IDF reference and diagnostics |
| P11-D | In progress | RoBERTa training complete; detailed rare-label score analysis next |
| P11-E | Planned | Synthetic-text generation protocol, provenance, review, deduplication, quality pilot |
| P11-F | Planned | Controlled real-only vs real-data resampling vs synthetic augmentation experiments |
| P11-G | Planned | Model selection and backend/frontend integration; cleanup of confirmed obsolete files |
| P12 | Planned | Reproducibility, documentation, demos, model/data cards, packaging |

Synthetic data will be treated as an **experimental intervention**, not as automatically correct labels or guaranteed improvement. Generated data must not use held-out evaluation texts as prompts. Final comparisons will retain human-annotated held-out evaluation and explicitly report limitations.

## Historical foundation: DAIR.AI Emotion (P0–P10)

The original laboratory used [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion), `split` configuration, with 16,000 train / 2,000 validation / 2,000 test and six labels. Its reported **test** metrics remain historical results, not scores comparable to GoEmotions:

| Historical model | Test accuracy | Test macro F1 | Test weighted F1 |
|---|---:|---:|---:|
| TF-IDF + logistic regression (`baseline_tfidf_dair_v001`) | 0.8320 | 0.7301 | 0.8208 |
| Manual NumPy MLP (`numpy_mlp_dair_v002`) | 0.8770 | 0.8058 | 0.8742 |
| TextCNN (`textcnn_dair_v001`) | **0.8965** | **0.8510** | **0.8967** |

The historical application implemented live TextCNN prediction/activation-span inspection and artifact-backed cross-model comparison and benchmark disagreement analysis. This code lives under `DAIR-MODEL-OLD/`; its old imports may no longer work after restructuring. See `DAIR-MODEL-OLD/project_state.md` for the historical snapshot. The original DAIR benchmark and GoEmotions are **different classification tasks**; comparing their numerical F1 values directly is not scientifically valid.

## Repository layout

```text
DAIR-MODEL-OLD/        Archived DAIR ML implementation and former project state
ml/src/               New multi-label data and model code
ml/scripts/           New ingestion, audits, training, analysis scripts
ml/configs/           Versioned new experiment configurations
ml/artifacts/         Local generated models, predictions, metrics and logs
data/raw/             Local versioned raw snapshots and manifests
data/processed/       Local prepared GoEmotions splits and manifests
reports/goemotions/   Generated audits and experiment diagnostics
tests/ml/             New ML tests; legacy test files are retained
backend/              Existing DAIR-era FastAPI service (migration pending)
frontend/             Existing React application (migration pending)
docs/                 Active project state and cumulative decision log
```

Generated datasets and trained weights are **not** intended to be committed to Git. Versioned configs, scripts, reports, manifests and focused tests are intended to be tracked. Preserve data provenance and file hashes so artifacts can be regenerated and verified. Large raw snapshots should be obtained from their pinned upstream revisions instead of Git.

## Local development

Primary development environment: Windows PowerShell, Python virtual environment `.venv`, NVIDIA CUDA GPU for RoBERTa training.

```powershell
cd C:\TmotionLab
.\.venv\Scripts\Activate.ps1

# Verify source schemas and local data snapshots
python -m ml.scripts.verify_goemotions
python -m ml.scripts.analyze_goemotions_token_lengths

# Test the active multi-label modules
pytest -q tests/ml/test_goemotions_baseline.py
pytest -q tests/ml/test_goemotions_dataset.py
pytest -q tests/ml/test_goemotions_roberta.py

# Train the baseline or RoBERTa using their versioned configs.
# These produce sizeable local artifacts; do not rerun an existing
# experiment ID without first inspecting its output directories.
python -m ml.scripts.train_goemotions_tfidf
python -m ml.scripts.train_goemotions_roberta
```

**Reproduction prerequisite:** The ingestion scripts and manifests use pinned Hugging Face dataset revisions; the RoBERTa tokenizer/model also uses a pinned revision. Follow the scripts and configs for exact library/runtime expectations. The project has not yet completed a clean-machine reproduction guide or an integrated post-migration application test suite.

## Evaluation principles and limitations

- Always distinguish **training**, **validation**, and **test** scores. The GoEmotions held-out test evaluation is intentionally deferred while the research protocol is being finalized.
- A multi-label prediction is a *set* of labels: sigmoid scores do not sum to one, and the model may select zero or several labels.
- Benchmark disagreement is not necessarily an unambiguous semantic mistake; annotation ambiguity and alternative plausible emotions exist.
- Raw annotation votes, per-label scores, and generated labels are not psychological ground truth. The software is not a diagnostic tool.
- Preserving official evaluation splits supports benchmark comparability, but the documented validation/test text overlap limits strict independence claims.
- Historical TextCNN activation spans are architecture-specific and must not be relabeled as RoBERTa causal explanations.

## Documentation

See `docs/project_state.md` for current milestones, `docs/decision_log.md` for architectural and scope decisions, and `DAIR-MODEL-OLD/project_state.md` for the earlier full-stack DAIR implementation.
