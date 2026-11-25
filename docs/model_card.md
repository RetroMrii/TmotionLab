# Model card — roberta_goemotions_synthetic10_v001

## Intended use

Interactive English text emotion classification and educational research. This
is a 28-label multi-label classifier (27 GoEmotions emotions plus neutral), not a
chatbot, clinical assessment, personality inference or causal explanation system.
Neutral may co-occur with other labels. Independent sigmoid model probabilities
are not calibrated confidence and do not sum to one.

## Model and provenance

- Encoder: RoBERTa-base, revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- Run: OPT-010 synthetic10, seed 42, standard BCE, real rare4x sampling and 10%
  synthetic draws, four epochs / 5,408 optimizer steps.
- Source: `ml/artifacts/optimization/opt010/synthetic10/best_model`.
- Frozen identity: [serving manifest](../backend/model-manifests/goemotions_synthetic10_v001.json).
- Canonical labels: [GoEmotions manifest](../data/processed/goemotions_v001/manifest.json).
- Synthetic corpus: OPT-009, 41,927 retained of 42,000 candidates. Review covered
  samples; residual annotation errors remain possible.
- Dataset revision: `google-research-datasets/go_emotions` at
  `add492243ff905527e67aeb8b80c082af02207c3`. Upstream dataset/model licenses and
  synthetic provider terms remain applicable; this promotion grants no new
  redistribution permission and creates no public artifact distribution.

## Evaluation and selection

Selected by validation macro AP within the controlled OPT-010 mixtures, with
macro/micro F1 supporting the decision. Exact saved validation scores:

| Metric | Value |
|---|---:|
| Macro average precision | 0.5183329791101571 |
| Macro F1 at 0.25 | 0.5208970777876761 |
| Micro F1 at 0.25 | 0.5968855589833995 |

Validation contains 5,426 examples. Micro F1 is slightly below OPT-010 control.
Large grief/pride AP gains have very small supports. The earlier weighted-BCE
candidate had higher AP but lower F1; it was not the supervisor's selected
standard-BCE model. See [optimization history](optimization_history.md).

There is **no independent post-selection test evaluation** for this model.
Selection reused validation and only one seed was trained per OPT-010 mode.
Multi-seed replication and selection bootstrap analysis were deliberately
skipped. P11's rare4x held-out metrics do not evaluate this checkpoint.

## Serving and artifact boundary

API `POST /v1/goemotions/predict`, schema 1, API release 0.5.0. Input: nonblank
English text, at most 5,000 characters; tokenization truncates at 48 tokens
including special tokens and returns an explicit truncation flag. Detections
use `score >= 0.25`, a fixed global policy inherited from the earlier validation
policy, not the OPT-008/011 tuned thresholds. All 28 scores are always returned
in canonical order; top five are separately ranked with stable label-ID ties.

Provision these five exact local files separately: `config.json`,
`model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `training_args.bin`.
Their total is 502,259,227 bytes; weights alone are 498,692,800 bytes. No public
download or Git LFS packaging exists. `training_args.bin` is hash-checked only,
not deserialized by serving. Verification fails closed on missing, changed or
additional files. Both mapping directions must match the canonical 28 labels.
Old rare4x weights and all source experiments remain untouched.

CPU is the default, CUDA optional. Eval/inference mode disables gradients.
One model loads and warms per process; requests serialize under a predictor lock.
More workers mean more resident model copies. Missing/corrupt artifacts yield
degraded health and HTTP 503 prediction responses, without blocking static
research or available DAIR saved-artifact routes.

## Limitations

Short English Reddit-derived training context, domain shift, annotation ambiguity,
unverified synthetic labels, tiny rare-label supports and documented validation/
test text overlap limit claims. Long text loses context after truncation. The
artwork and color families are design choices, not model explanations. No
TextCNN activation method is applied to RoBERTa. Provisioning, deployment and
independent evaluation remain separate supervisor decisions.
