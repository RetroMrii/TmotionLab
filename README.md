# TmotionLab

**An interactive emotion-analysis application and a reproducible study of fine-grained emotion classification.**

TmotionLab combines a React reading workspace with a locally served RoBERTa model trained on **GoEmotions: 27 emotions plus neutral**. Explore a passage, inspect its independent emotion scores, and examine the experiments behind the selected model.

## Explore the application

- **Analyze** — submit English text to the live model; inspect detected labels, top-ranked scores, all 28 scores, and the tokens actually analyzed. Multiple emotions, neutral alongside other emotions, or no detections are valid results.
- **Reading Studio** — a responsive writing workspace with light/dark themes, related emotion color families, accessible keyboard navigation, and a continuously flowing SVG **Emotion Signal**. The ribbons are decorative artwork, not neural activity or model attribution.
- **Research Overview** — the study, experimental controls, findings and scientific limitations, with downloadable source evidence.
- **Experiment Explorer** — compare 12 validation runs, three-seed means and sample standard deviations, rare-label results, and the separately reported final held-out evaluation.
- **Historical Compare** — inspect saved predictions from three earlier models on the **DAIR six-emotion dataset**. Enter through “Compare historical benchmarks” beneath Analyze, or visit `/compare`. This is an archived benchmark instrument, not another GoEmotions predictor.

The application is complete and verified locally. This repository does not provide a hosted website or a public download of the selected model weights.

## Model and research

The controlled study compares **four sampling strategies across seeds 42, 123 and 456**:

| Strategy | Training intervention |
|---|---|
| Uniform | Real examples sampled uniformly with replacement |
| Real 4× | Increased sampling weight for examples containing targeted rare emotions |
| Synthetic | Targeted draws from 299 generated examples mixed with real data |
| Hybrid | A 50/50 mixture of the real 4× and synthetic sampling schedules |

All runs use the same RoBERTa base revision, four epochs and 5,408 optimizer steps. Checkpoints are selected by validation macro average precision; each run's global threshold is selected on validation. Three seeds support descriptive comparisons, not a claim of statistical significance.

The selected model is **`roberta_goemotions_rare4x_v001`**, seed **42**, with a frozen global sigmoid threshold of **0.25**. Real-data 4× oversampling was chosen for competitive performance without depending on unverified synthetic labels. Hybrid achieved the highest mean validation macro F1, but its small advantage did not establish reliable superiority.

### Final held-out evaluation

The selected model was frozen before evaluation on **5,427 test examples**. These are saved results, not metrics recomputed during installation, serving or frontend builds.

| Metric | Score | Definition |
|---|---:|---|
| Macro F1 | 0.523654 | Unweighted mean of per-label F1 across 28 labels |
| Micro F1 | 0.600000 | F1 from pooled true positives, false positives and false negatives across labels |
| Macro average precision | 0.514646 | Unweighted mean of per-label average precision from the continuous scores; not precision at threshold 0.25 |
| Exact-match accuracy | 0.420490 | Fraction of examples whose entire predicted label set equals the reference set |
| Example-based F1 | 0.612677 | Mean F1 computed separately for each example's predicted/reference label sets |

F1 uses zero for undefined divisions, as in the evaluation implementation. See the [frozen evaluation summary](reports/goemotions/final_heldout_test_v001.json), [model freeze](reports/goemotions/final_model_freeze_v001.json) and [research report](reports/goemotions/goemotions_research_final_v001.md).

### Data and provenance

The processed training view contains **43,252 examples**, after removing 158 original training rows with normalized-text overlap against validation or test. Official validation (5,426) and test (5,427) membership is preserved. Eleven normalized texts still overlap validation and test; exact-text checks do not rule out near-duplicates.

- Dataset: `google-research-datasets/go_emotions`, revision `add492243ff905527e67aeb8b80c082af02207c3`.
- Base model: `FacebookAI/roberta-base`, revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- [Dataset manifest and canonical label order](data/processed/goemotions_v001/manifest.json).
- [Experiment configuration](ml/configs/roberta_goemotions_multilabel_real_v001.json).

Historical DAIR Compare uses saved TF-IDF/logistic-regression, NumPy MLP and TextCNN predictions. Its six-label single-label metrics are **not directly comparable** with GoEmotions multi-label results. Archived live `/predict` and `/interpret` APIs intentionally return unavailable responses; the historical saved-artifact APIs remain functional.

## Architecture

```text
React + TypeScript + Vite
  ├─ Analyze → FastAPI → frozen local GoEmotions RoBERTa (PyTorch/Transformers)
  ├─ Research / Experiment Explorer → generated, hash-verified static evidence
  └─ Historical Compare → FastAPI → saved DAIR comparison artifacts
```

The primary API is **`POST /v1/goemotions/predict`** (FastAPI API version 0.4.0). Input must be a nonblank string of at most 5,000 characters. Inference uses at most **48 tokens, including special tokens**, and reports truncation explicitly. The API returns canonical scores for all 28 labels, detections at `score >= 0.25`, and the top five ranked scores. Scores are independent sigmoid outputs, not calibrated confidence or a probability distribution summing to one.

Real inference starts immediately. Analyze maintains a minimum 1.5-second presentation sequence, then reveals the actual result. Editing, clearing, switching presets or navigating cancels the frontend request/reveal sequence and prevents stale results. Client cancellation does not promise interruption of an already-running server forward pass.

## Run locally

Commands below are **PowerShell**, run from the repository root. Node **26.3.0 / npm 11.16.0** and Python **3.14.5** were used for local verification. Python dependencies in `requirements.txt` are not pinned, so a fresh Python install is not guaranteed to reproduce the verified environment. The current local environment also has a `datasets`/`huggingface-hub` version constraint conflict; it is not a portable lockfile. No dependency upgrade is required by the frontend-only workflow.

### Frontend and research, without model weights

```powershell
npm --prefix frontend ci
npm --prefix frontend run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`. `predev` generates research assets from the 38 committed, hash-pinned originals. Research remains readable without the API. Live Analyze requires the separately provisioned backend/model; historical Compare requires the backend's saved DAIR artifacts.

To produce and inspect a static build:

```powershell
npm --prefix frontend run build
npm --prefix frontend run preview -- --host 127.0.0.1
```

The preview command prints its local URL. The build generates research assets, verifies source/output integrity, typechecks and bundles to `frontend/dist/`.

### Backend and live Analyze

Use an existing compatible Python environment, or create one and install the declared requirements (unlocked bootstrap, not a certified exact environment):

```powershell
# Only when creating a new environment:
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Resolve any installation/version conflicts before treating a new environment as verified. The repository does not currently ship an environment lock or a model download script.

Provision the original frozen files separately at this exact repository-relative directory:

```text
ml/artifacts/models/roberta_goemotions_rare4x_v001/best_model/
  config.json
  model.safetensors
  tokenizer.json
  tokenizer_config.json
  training_args.bin
```

The five files total **502,259,227 bytes (approximately 479 MiB)**. Obtain the exact bundle from its custodian; there is no published download location in this repository. Do not substitute base RoBERTa weights, remove `training_args.bin`, add files inside `best_model`, or alter the freeze to bypass verification. All five files are excluded from ordinary Git.

The verifier also requires the committed `reports/goemotions/final_model_freeze_v001.json` and `data/processed/goemotions_v001/manifest.json`. It checks exact file SHA-256 hashes, model identity, seed, threshold, manifest hash and both label mappings before loading.

```powershell
.\.venv\Scripts\python.exe -c "from backend.app.services.goemotions import verify_artifacts; print('Verified labels:', len(verify_artifacts()))"
$env:TMOTIONLAB_DEVICE = 'cpu'
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Run the frontend in a second terminal. The API defaults to `http://127.0.0.1:8000`, allows the localhost/127.0.0.1 Vite origins, and uses CPU by default. `TMOTIONLAB_DEVICE=cuda` or `auto` requests CUDA with CPU fallback during initialization. Backend `.env` files are not loaded automatically: export variables before starting Uvicorn. For a different API origin, set `VITE_API_BASE_URL` before frontend dev/build and configure `TMOTIONLAB_CORS_ORIGINS` on the backend.

If model initialization fails, the API starts in a degraded state: `/health` reports `goemotions_loaded: false` and Analyze returns HTTP 503. `/health` itself still returns HTTP 200. Static research remains available. Historical APIs can work independently when their saved artifacts load. They read the comparison CSV, two error/confusion CSVs and three metrics JSON files in `DAIR-MODEL-OLD/ml/artifacts/`; they do not load the archived model weights.

## Research generation and verification

[`frontend/research-sources.lock.json`](frontend/research-sources.lock.json) records the exact **38 original scientific files**: aggregate reports, manifests/configuration, twelve validation metric files and twelve provenance files. These are versioned evidence, distinct from ignored internal engineering reports.

```powershell
npm --prefix frontend run research:generate
npm --prefix frontend run research:check
```

Generation verifies the source hashes and recreates `frontend/src/data/research.json` plus exact download copies under `frontend/public/research/sources/`. These generated outputs are ignored by Git. `research:check` is read-only and requires the generated outputs to exist; normal dev/build generates them first. Do not use `--pin` or `--add-sources` to repair an integrity failure.

The narrow `.gitattributes` entries preserve the pinned originals byte-for-byte,
including their existing line endings, across Git checkouts on different systems.

The frontend research build needs neither weights nor raw datasets, makes no inference calls and never reruns held-out evaluation. Source downloads expose scientific metadata and aggregate results, not individual GoEmotions test predictions. Historical DAIR Compare separately exposes saved example text and predictions by design.

## Tests

With frontend dependencies installed and research generated:

```powershell
npm --prefix frontend run lint
npm --prefix frontend run build
# Start the API on port 8000 and Vite on port 5173 first; installed Chrome required:
npm --prefix frontend test
```

The Playwright suite uses system Chrome, both themes and desktop/mobile widths. It starts a separate QA Vite service on port 5174 and uses port 65530 as an unreachable API origin; keep those QA ports free. Coverage includes live inference, request cancellation/stale results, navigation, scrolling, all emotion colors, ribbon motion, research interactions and automated accessibility scans. Its geometry baseline is a committed test fixture. Test screenshots/logs are ignored.

Backend regression requires the exact frozen model and historical artifacts. The validation-compatibility test additionally needs these local-only files, which are not distributed through Git:

```text
data/processed/goemotions_v001/validation.jsonl
ml/artifacts/predictions/roberta_goemotions_rare4x_v001_validation.jsonl
```

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/backend -p no:cacheprovider
# With the backend running:
.\.venv\Scripts\python.exe -m backend.scripts.smoke_api
# Optional: compare 16 saved validation examples; requires both local files above:
.\.venv\Scripts\python.exe -m backend.scripts.verify_goemotions_serving
```

These serving checks do not reevaluate the held-out test set. The archived DAIR test suite depends on historical imports and is outside the active backend suite. Some ML tests require local data/research artifacts; a bare checkout is sufficient for the static research build, not every model-dependent test.

## Scientific and interaction limitations

- Emotions are model outputs about text, not diagnoses or facts about a person's psychological state. The application is not a clinical tool.
- Rare labels have very small evaluation supports. Three-seed comparisons and one held-out evaluation do not establish broad robustness or statistical superiority.
- The model processes English text and truncates to 48 tokens; longer passages can lose essential context. Dataset/domain and annotation biases remain.
- Official validation/test text overlap limits strict independence claims. Synthetic secondary labels were not independently semantically verified; the selected model uses real data.
- Color families and animated ribbons are presentation choices, not psychological categories, calibrated intensity or causal explanations.
- The current motion policy continues animations despite system reduced-motion preferences. Automated accessibility checks do not establish complete accessibility conformance; physical-device performance and cross-browser behavior have limitations.

## Repository guide

| Path | Contents |
|---|---|
| `frontend/` | Reading Studio, routes, research exporter, source lock and browser tests |
| `backend/` | FastAPI routes, frozen-model serving and historical artifact access |
| `ml/src/`, `ml/scripts/`, `ml/configs/` | Data preparation, controlled experiments and reproducibility code |
| `reports/goemotions/` | Narrowly versioned scientific evidence; other local outputs remain ignored |
| `data/` | Versioned manifests; dataset rows remain local |
| `DAIR-MODEL-OLD/` | Historical six-emotion implementation and saved serving evidence |
| `tests/` | Backend and ML regression tests |
| `docs/` | API reference and project documentation |

See the [frontend guide](frontend/README.md), [API contract](docs/api_contract.md) and [research report](reports/goemotions/goemotions_research_final_v001.md). Local data, model weights, caches and internal engineering reports are intentionally outside the source checkpoint.
