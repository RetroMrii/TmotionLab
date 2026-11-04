# TmotionLab — GoEmotions Research Report

**Study:** P11 — Rare-Emotion Sampling and Synthetic Augmentation

**Status:** Completed exploratory research; primary model evaluated on held-out test.

## Executive summary

This study investigated whether targeted synthetic examples improve recognition of rare emotions in multi-label text classification, relative to real-data oversampling.

Across three training seeds and four sampling strategies, all rare-label interventions improved validation macro F1 relative to uniform sampling.

Hybrid sampling achieved the highest mean validation macro F1, but its advantage over real-data oversampling was too small to establish reliable superiority.

Real 4× oversampling was selected as the primary model under the predefined pragmatic decision: prefer the simpler human-annotated-data approach unless hybrid demonstrates a meaningful benefit.

## 1. Dataset and task

- Dataset: GoEmotions simplified.
- Task: 28-label multi-label emotion classification.
- Processed training rows: 43,252.
- Validation rows: 5,426.
- Held-out test rows: 5,427.
- Training/validation normalized-text overlap was addressed during dataset preparation.
- The official test labels were excluded from training, model selection and threshold tuning.

## 2. Baseline and interventions

| Strategy | Training intervention |
|---|---|
| Uniform | Real examples sampled uniformly with replacement |
| Real 4× | Increased sampling weight for examples containing targeted rare emotions |
| Synthetic | Targeted draws from 299 generated examples mixed with real data |
| Hybrid | 50/50 mixture of real 4× and synthetic sampling schedules |

The synthetic candidates were produced using GPT-6.1 Terra. The original pilot contained 300 candidates; one confirmed near-duplicate was excluded.

Individual human semantic annotation was waived for this dataset as a one-time research decision. Generator-proposed secondary labels were therefore not independently verified.

## 3. Experimental controls

- Base model: RoBERTa-base, pinned source revision.
- Seeds: 42, 123 and 456.
- Training: 4 epochs and 43,252 draws per epoch.
- Optimization: 5,408 steps per run.
- Checkpoint selection: validation macro AP.
- Global classification threshold: selected from a fixed validation grid per run.
- Sampling schedules: deterministic and audited.
- Test split: excluded from strategy development.

## 4. Three-seed validation results

| Strategy | Macro F1 mean ± SD | Macro AP mean ± SD | Micro F1 mean |
|---|---:|---:|---:|
| uniform | 0.4678 ± 0.0065 | 0.4909 ± 0.0172 | 0.5971 |
| rare4x | 0.5173 ± 0.0048 | 0.5160 ± 0.0026 | 0.5974 |
| synthetic | 0.5147 ± 0.0113 | 0.5085 ± 0.0065 | 0.5972 |
| hybrid | 0.5200 ± 0.0048 | 0.5167 ± 0.0053 | 0.5984 |

Statistics are descriptive. Three seeds do not establish statistical significance or reliable superiority between closely performing methods.

## 5. Four-target rare-emotion analysis

| Strategy | Four-target mean F1 ± SD | Four-target mean AP |
|---|---:|---:|
| uniform | 0.0000 ± 0.0000 | 0.1664 |
| rare4x | 0.3645 ± 0.0277 | 0.3561 |
| synthetic | 0.3401 ± 0.0813 | 0.3067 |
| hybrid | 0.3772 ± 0.0145 | 0.3542 |

### Per-label mean validation F1

| Emotion | Uniform | Real 4× | Synthetic | Hybrid |
|---|---:|---:|---:|---:|
| grief | 0.0000 | 0.3742 | 0.2741 | 0.3544 |
| nervousness | 0.0000 | 0.2585 | 0.4069 | 0.3708 |
| pride | 0.0000 | 0.5693 | 0.5082 | 0.5328 |
| relief | 0.0000 | 0.2561 | 0.1711 | 0.2508 |

Hybrid improved nervousness F1 relative to real 4× in all three seeds, but its differences on the other rare labels were mixed.

The rare-label evaluation supports are small; conclusions about individual emotions therefore remain uncertain.

## 6. Final model selection

- Selected: `roberta_goemotions_rare4x_v001`.
- Seed: 42 (fixed reference seed).
- Threshold: 0.25, chosen on validation.
- Selection rationale: competitive performance without requiring unverified synthetic labels.
- Hybrid retained as the strongest research alternative, not the primary deployed checkpoint.

## 7. Held-out test results

| Metric | Score |
|---|---:|
| Macro F1 | 0.5237 |
| Micro F1 | 0.6000 |
| Macro AP | 0.5146 |
| Exact-match accuracy | 0.4205 |
| Example-based F1 | 0.6127 |

Test examples: **5,427**.

### Rare-label held-out performance

| Label | F1 | Support |
|---|---:|---:|
| grief | 0.6667 | 6 |
| nervousness | 0.2444 | 23 |
| pride | 0.5000 | 16 |
| relief | 0.2727 | 11 |

The final test was used once after freezing the primary model and threshold. No test-time tuning was performed.

## 8. Limitations

1. Only three paired training seeds were used.
2. Rare-label validation and test supports are small.
3. Synthetic labels were not independently reviewed.
4. Thresholds and interventions were developed using the same validation split.
5. Models may behave differently on informal real-world inputs outside the benchmark.
6. The held-out evaluation covers the selected real 4× model, not all four strategies.

## 9. Conclusions

Targeted exposure to rare-emotion examples consistently improved validation macro F1. Synthetic generation supplied useful training signal, particularly for nervousness, but did not demonstrate reliable overall superiority to real-data oversampling.

The selected real 4× model obtained a held-out macro F1 of 0.5237, with micro F1 0.6000 and exact-match accuracy 0.4205.

The project can now transition to artifact retention and application integration without altering the completed research results.

## 10. Evidence and reproducibility

The following source checksums identify the exact results used to construct this report:

| Source | SHA-256 |
|---|---|
| `reports\goemotions\multiseed_robustness_v001.json` | `e217bf39024b1394ceb736e66f25a9a9f9f4b9f2fd7855f007ea58f087b7cb8c` |
| `reports\goemotions\multiseed_rare_label_analysis_v001.json` | `c67dd4bb596061e980ab20477270b160b58f70e226d648355ebe6be3d45c82cd` |
| `reports\goemotions\final_heldout_test_v001.json` | `e640be4a23a77b5755ca56520af5125715053b49dc97eaf7e9227da56916f3df` |
| `reports\goemotions\final_model_freeze_v001.json` | `01868a7770a1a8f3be4ca75d5b4518b36305376d422b134dbeee1d4224bc86a1` |

The source artifacts additionally retain per-run metrics, prediction hashes, selected thresholds, sampling audits and experiment provenance.
