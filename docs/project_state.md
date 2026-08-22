# Current TmotionLab State

**Current phase:** P2_dataset_pipeline

## Completed

- Project renamed to TmotionLab.
- Core portfolio thesis changed to Predict + Compare + Interpret + Error Lab.
- GoEmotions / multi-label work moved to advanced experiments.
- README separated from the internal project plan.
- Base repository directories and Python/test skeleton created.
- DAIR.AI Emotion upstream dataset contract verified against current Hugging Face/GitHub metadata.
- Selected `dair-ai/emotion` configuration `split` with fixed 16k/2k/2k train/validation/test boundaries.
- Added `ml/configs/dair_ai_emotion.yaml` with verified columns, labels, sizes, license/usage note, and pinned revision.
- Added `ml/scripts/inspect_dataset.py` with contract validation and optional JSON inspection output.
- Added `docs/dataset_notes.md` documenting verified metadata and remaining local measurements.
- Added static dataset-contract tests; 2 tests pass.

## Current files changed

- `ml/configs/dair_ai_emotion.yaml`
- `ml/scripts/inspect_dataset.py`
- `tests/ml/test_dataset_config.py`
- `docs/dataset_notes.md`
- `docs/decision_log.md`
- `docs/project_state.md`

## Current decisions

See `docs/decision_log.md`.

## Verification performed

```text
python -m py_compile ml/scripts/inspect_dataset.py                  PASS
python ml/scripts/inspect_dataset.py --help                        PASS
python ml/scripts/inspect_dataset.py --config ml/configs/dair_ai_emotion.yaml --validate-config-only  PASS
pytest -q tests/ml/test_dataset_config.py                           2 passed
```

## Current errors or risks

- Live Hugging Face dataset loading has not been executed in this runtime because the `datasets` package is absent and outbound package installation is blocked by network/DNS restrictions.
- Exact class counts, duplicate rates, text-length statistics, and empty-text counts are not yet locally measured. The inspection script will calculate class counts and empty-text counts once the dataset is loaded.
- Dependency versions remain intentionally unpinned until the target development environment is verified.
- No processed dataset, model metrics, or model artifacts exist yet.

## Next exact step

On the development machine, install dependencies and run the live dataset contract inspection:

```bash
python -m pip install -r requirements.txt
python ml/scripts/inspect_dataset.py --config ml/configs/dair_ai_emotion.yaml --output ml/artifacts/metrics/dair_ai_emotion_inspection.json
```

Only after `[OK] Dataset contract validated` should `ml/scripts/prepare_dataset.py` be implemented/run to create processed splits with stable example identifiers.

## Do not forget

- Preserve upstream train/validation/test boundaries.
- Do not remap the verified label order silently.
- Processed examples must receive stable IDs so later models can be aligned exactly in Compare and Error Lab.
- TextCNN interpretation is architecture-native model evidence, not causal psychological explanation.
Current phase: P6_model_comparison_lab — complete

Completed:
- aligned three-model comparison
- pairwise and all-model agreement analysis
- TextCNN fix/regression cohorts
- shared-error cohort
- three-way disagreement cohort
- generated model-comparison report

Next exact step:
P7_textcnn_interpretation_lab
Current phase: P7_textcnn_interpretation_lab — complete

Completed:
- TextCNN activation extraction
- exact token-span reconstruction
- padding-overlap detection
- held-out interpretation artifact
- aggregate filter/span analysis
- generated interpretation report

Next exact step:
P8_error_lab
Current phase: P8_error_lab — complete

Completed:
- canonical TextCNN benchmark-disagreement dataset
- confusion-pair analysis
- high-score disagreement analysis
- shared cross-model disagreement cohort
- TextCNN regression cohort
- partial interpretation metadata integration
- generated Error Lab report
- benchmark-relative evaluation caveat

Deferred:
- manual adjudication
- human-reviewed error taxonomy
- dataset relabeling analysis
- calibration analysis

Next exact step:
P9_fastapi_backend
