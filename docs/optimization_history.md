# GoEmotions optimization and model promotion

Current decision: **2026-11-23**, supervisor-approved promotion of OPT-010
`synthetic10` as `roberta_goemotions_synthetic10_v001`. Earlier no-promotion
decisions remain historical records. The production threshold stays global 0.25.

## Scope and evidence

The following scores were independently reproduced from eight saved validation
prediction files, aligned by example ID and gold labels over all **5,426** official
validation examples. No test split was evaluated. Exact aggregate values,
per-label statistics and source SHA-256 hashes are published in
[`frontend/src/data/optimization.json`](../frontend/src/data/optimization.json).
The verification program is
[`backend/scripts/verify_model_promotion.py`](../backend/scripts/verify_model_promotion.py).

| Study / encoder | Configuration | Macro AP | Macro F1 | Micro F1 |
|---|---|---:|---:|---:|
| OPT-007 RoBERTa | Standard BCE | 0.510322 | 0.516799 | 0.600949 |
| OPT-007 RoBERTa | Weighted BCE | 0.520105 | 0.509201 | 0.587179 |
| OPT-010 RoBERTa | Control | 0.503279 | 0.513615 | 0.598228 |
| OPT-010 RoBERTa | **Synthetic10, promoted** | **0.518333** | **0.520897** | **0.596886** |
| OPT-010 RoBERTa | Synthetic25 | 0.514961 | 0.510986 | 0.592571 |
| OPT-010 RoBERTa | Synthetic40 | 0.505621 | 0.516074 | 0.591644 |
| OPT-012 DeBERTa-v3-base | Control | 0.500245 | 0.509003 | 0.591022 |
| OPT-012 DeBERTa-v3-base | Synthetic10 | 0.509283 | 0.516394 | 0.586501 |

These are fixed-threshold (0.25) F1 scores. Macro AP is the mean of 28 per-label
average precisions using continuous scores, independent of a detection threshold.
Macro F1 gives each label equal weight; micro F1 pools label decisions and gives
frequent labels greater influence. None measures calibrated confidence.

## Earlier foundation

The original DAIR work progressed through TF-IDF/logistic regression, a manual
NumPy MLP and PyTorch TextCNN, with six-class comparison and convolutional
activation inspection. Those artifacts and IDs remain under `DAIR-MODEL-OLD`.
GoEmotions adds overlapping labels, 27 emotions plus neutral, and rare-label
research. P11 compared uniform, real 4× oversampling, small synthetic and hybrid
sampling over seeds 42/123/456 (12 runs). The selected historical rare4x seed-42
model had validation macro AP **0.514552002815609**, macro F1
**0.5158028109555619**, micro F1 **0.5987997658079626**. The original approximation
0.5146 is not an additional measured result. Its later held-out evaluation remains
attributed exclusively to rare4x. P12 introduced the separate GoEmotions v1 API.

## OPT-007 and OPT-008: objective and thresholds

Positive-class-weighted BCE improved ranking AP over standard BCE but lowered
both thresholded F1 measures. It is not the promoted training objective. The
supervisor chose the standard-BCE synthetic candidate while accepting the
ranking/F1 tradeoff; weighted BCE's higher AP is disclosed rather than omitted.

OPT-008 used five-fold out-of-fold per-label thresholds. Standard BCE reached
macro F1 **0.5208304347864707** and micro F1 **0.5988334413480233**. These are tuned
OOF results, not the fixed 0.25 metrics above and not the production policy.
Cross-fitting threshold estimation does not undo checkpoint selection on the
same validation set. Source: local
`reports/goemotions/optimization/opt008_weighted_comparison.json`.

## OPT-009: synthetic corpus

42,000 candidates yielded 41,927 retained examples after excluding 73 unique
examples. The local dataset is
`data/processed/synthetic_goemotions_opt009_v001/train.jsonl`, with `manifest.json`
in the same directory. Reasons overlap: label disagreement 46, synthetic semantic
similarity 24, unsupported primary label 19, ambiguous annotation 6, verifier
review request 3, duplicate text 2. Do not sum those reason counts.

Supervisor-reviewed samples were acceptable. This does not mean all 41,927 rows
were individually human-annotated or semantically verified. Synthetic label
quality remains a limitation. The immutable export manifest still records its
historical `training_approval: false`; subsequent training and this explicit
promotion approval supersede that workflow state without rewriting provenance.

An earlier diagnostic file named `optimization/opt009_ap_robustness.json` uses
the same OPT-009 identifier for weighted-BCE bootstrap work (300 paired
replicates). It is separate from synthetic corpus generation. No bootstrap or
multi-seed replication was run for the current OPT-010 promotion decision.

## OPT-010: controlled synthetic mixtures

Four runs used RoBERTa-base, real rare4x sampling, standard BCE, seed 42, four
epochs, 5,408 optimizer steps and the same validation split. Synthetic fractions
were 0%, 10%, 25%, 40% of scheduled training draws, not proportions of unique
dataset rows. Synthetic10 used 38,927 real and 4,325 synthetic draws per epoch,
with replacement. The real base has 43,252 examples after overlap cleanup.

Synthetic10 led this controlled comparison in macro AP and macro F1, but had
slightly lower micro F1 than control. It is a single-seed validation selection,
not a demonstrated improvement on unseen data or a universal winner across all
earlier objectives. The selected checkpoint remains in its original directory;
production uses a separately versioned hash manifest, without copying weights.

## OPT-011: per-label tradeoffs

Saved control and synthetic10 predictions align on all 5,426 examples. Macro AP
improved by approximately 0.01505. Selected per-label AP differences:

| Emotion | Synthetic10 minus control AP |
|---|---:|
| Grief | +0.2125 |
| Pride | +0.1479 |
| Remorse | +0.0671 |
| Embarrassment | +0.0486 |
| Relief | +0.0422 |
| Nervousness | −0.0847 |
| Disappointment | −0.0536 |
| Confusion | −0.0282 |

Grief and pride have only 13 and 15 positive validation examples. The complete
28-label table is available in Research Overview and the aggregate JSON. Five-fold
OOF thresholding did not recover the micro F1 regression: control approximately
0.515625 macro / 0.594494 micro; synthetic10 0.522491 macro / 0.588523 micro.
The original `reports/goemotions/opt011/report.md` and `analysis.json` remain
unchanged, including their historical no-promotion recommendation.

## OPT-012: alternative encoder

The initial DeBERTa control attempt encountered FP16 gradient unscaling failure;
the retained `control_failed_fp16` artifacts record that attempt. The corrected
training script explicitly loaded and retained trainable parameters in FP32,
while allowing eligible AMP computation to use FP16. **Both final control and
synthetic10 completed four epochs and 5,408 steps.** Synthetic10 improved on
DeBERTa control in AP but remained below the selected RoBERTa candidate; its micro
F1 also decreased. “Unsuccessful” means inferior to the selection objective, not
failure of the completed synthetic10 training run.

## Reproducibility boundary

The public aggregate snapshot can be bundled without raw data or model weights.
To independently regenerate it, provision the original local validation rows,
all eight run directories (metrics, predictions, summaries), selected provenance,
synthetic corpus/manifest, OPT-008 report and OPT-011 analysis. Their exact paths
and hashes are in `source_sha256`; the synthetic data hash is in the provenance.

```powershell
.\.venv\Scripts\python.exe -m backend.scripts.verify_model_promotion
# Only for an explicitly reviewed new snapshot; never a repair for mismatches:
.\.venv\Scripts\python.exe -m backend.scripts.verify_model_promotion --write
```

Default verification is read-only. It compares saved metrics against recomputed
validation metrics, then compares published snapshots. It neither trains nor
performs new model inference nor reads the held-out test split. Raw examples and
row-level predictions are not published to the website. Original P11 research
generation continues independently with its unchanged 38 hash-pinned sources.
