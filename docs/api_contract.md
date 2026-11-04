# TmotionLab API 0.4.0

FastAPI exposes live OpenAPI at `/openapi.json` and Swagger at `/docs`.
`docs/openapi.json` is the checked-in schema snapshot. CORS defaults to
localhost/127.0.0.1 port 5173; configure exported `TMOTIONLAB_CORS_ORIGINS`
before startup. No runtime network download or synthetic fallback is permitted.

## GoEmotions v1

`POST /v1/goemotions/predict`, JSON `{"text":"I love this, thank you!"}`.
Text must be a string of 1–5000 characters and contain non-whitespace. Extra
parameters, including threshold overrides, are rejected. Input is preserved
exactly, including whitespace. The local tokenizer uses truncation, padding=True
and max_length=48, matching final evaluation (single-item requests need no pad).

The response includes:

| Field | Meaning |
| --- | --- |
| `schema_version`, `task` | `"1"`, `"multi_label"` |
| `model_id`, `dataset`, `seed` | Frozen rare4x model, `goemotions_v001`, 42 |
| `threshold`, `score_type` | Fixed 0.25, `sigmoid` |
| `scores` | All 28 `{label_id,label,score}` entries in frozen manifest order |
| `detected_labels` | Canonical-order labels whose scores are **>= 0.25**; may be empty |
| `ranked_scores` | Highest five scores, descending; ties use canonical label ID |
| `input.text` | Exact submitted text, retained even when truncation occurs |
| `input.tokens` | Actual analyzed token strings, including special tokens |
| `input.token_count`, `original_token_count` | Analyzed and original counts, including special tokens |
| `input.max_tokens`, `truncated` | 48 and whether original count exceeds 48 |
| `device`, `availability_notes` | Runtime device and any CUDA-to-CPU fallback notes |
| `score_note` | Calibration, neutral co-occurrence and interpretation limitations |

Scores need not sum to one. No top label is forced when none passes the threshold.
Neutral is learned and can coexist with other detections. Scores are not
calibrated confidence or evidence of a person's psychological state.

Model loading happens once per application lifespan. All five model-directory
files and the processed manifest must match the final freeze SHA-256 hashes.
The loaded model and manifest label mappings must agree. No raw `num_labels`
config field is required: Transformers derives it from `id2label`. Weights are
local only; eval/inference modes and deterministic algorithms are enabled.
CPU is the default; exported `TMOTIONLAB_DEVICE=cuda` or `auto` may use CUDA,
with CPU fallback if initialization/warm-up fails. Requests are serialized per
predictor to bound inference concurrency. A later inference failure returns 503;
it is not replaced with fabricated output. Restart after fixing startup failures.

## Compatibility and historical tools

| Route | API 0.4.0 behavior |
| --- | --- |
| `GET /health` | 200, `status=ok` when GoEmotions loads, otherwise degraded. Also reports `goemotions_loaded`, `textcnn_loaded=false`, `artifacts_loaded`, `model_id`, diagnostics. Check flags as well as HTTP status. |
| `GET /models` | Always discoverable; four registry entries, dataset/task/status and only currently functioning capabilities. GoEmotions has `live_predict` only. Archived entries have no live capabilities. Historical metric fields are null for GoEmotions. |
| `POST /predict` | Original DAIR six-label schema retained; valid requests return explicit 503 because the archived runtime is absent. Never aliases GoEmotions. |
| `POST /interpret` | Original TextCNN schema retained; explicit 503. No RoBERTa activation interpretation. |
| `GET /metrics` | Three historical DAIR test metric rows; no GoEmotions rows mixed in. |
| `GET /compare/{example_id}` | Saved DAIR comparison (e.g. `dair_test_001837`); unknown ID 404. |
| `GET /errors`, `/errors/confusions` | Existing DAIR filters, pagination, benchmark-relative notes and saved artifacts. |

Offline DAIR files load from `DAIR-MODEL-OLD/ml/artifacts/` without importing or
editing the archive. Missing archive files do not block Analyze; missing
GoEmotions files do not block valid offline tools. Missing offline data returns
503 rather than zero rows. The three historical models share a DAIR test split;
none shares the GoEmotions task/split. Their F1 values are not directly comparable.

Errors use `{ "error": { "code": "http_503", "message": "...", "details": null } }`.
Input/schema errors are 422 (`validation_error` for Pydantic validation; `http_422`
for whitespace-only text). Invalid inference numeric output returns JSON 503.
Detailed initialization errors, including hash mismatches, appear in server logs.

## Verification

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.verify_goemotions_serving
.\.venv\Scripts\python.exe -m pytest -q tests/backend -p no:cacheprovider --basetemp .venv/p12-test-temp
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
# In a second terminal, while the API runs:
.\.venv\Scripts\python.exe -m backend.scripts.smoke_api
npm --prefix frontend run dev -- --host 127.0.0.1
# With both services running:
npm --prefix frontend test
```

Verification reads only the first 16 existing validation predictions, allowing
the previously established 0.002 numerical tolerance; it writes no P11 artifacts.
Browser tests use installed Chrome and Playwright as a development dependency.
They distinguish live inference tests from isolated mocked edge-case tests.
