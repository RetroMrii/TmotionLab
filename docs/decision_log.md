# TmotionLab Decision Log

This file records project decisions that affect scope, architecture, evaluation, or development workflow. Newer accepted decisions supersede older conflicting documentation.

## Decision: Rename project to TmotionLab

**Date:** 2026-08-20  
**Status:** accepted  
**Context:** `TextEmotionLab` described the task clearly but had little project identity and read like a generic classifier name.  
**Decision:** Rename the project to **TmotionLab**.  
**Alternatives considered:** Keep TextEmotionLab; other descriptive emotion-classification names.  
**Reason:** TmotionLab is more distinctive and supports a laboratory/product identity rather than framing the work as only a classifier.  
**Impact:** Repository title, documentation, service identifiers, UI copy, and future package/deployment naming should use TmotionLab.

## Decision: Make model-behavior analysis the portfolio thesis

**Date:** 2026-08-20  
**Status:** accepted  
**Context:** A `text -> emotion label` demo is too common to be the primary differentiator for a strong ML-engineering portfolio project.  
**Decision:** The core TmotionLab experience will center on four layers: **Predict, Compare, Interpret, and Error Lab**.  
**Alternatives considered:** Focus mainly on adding more datasets; make full GoEmotions multi-label classification the major differentiator.  
**Reason:** Cross-model comparison, architecture-native interpretation, and systematic error analysis demonstrate stronger ML reasoning and engineering depth while reusing the same core modeling pipeline.  
**Impact:** Frontend pages, backend contracts, evaluation artifacts, harness tests, and development phases now include comparison, interpretation, and error-analysis work as first-class requirements.

## Decision: Defer GoEmotions and multi-label modeling

**Date:** 2026-08-20  
**Status:** accepted  
**Context:** The previous plan treated GoEmotions reduced/full taxonomy as a main late-stage milestone. This risks breadth before the core laboratory is polished and validated.  
**Decision:** Keep GoEmotions and multi-label support as advanced/stretch experiments after the single-label TmotionLab core is complete.  
**Alternatives considered:** Implement reduced GoEmotions immediately after the first TextCNN; make 27+neutral multi-label output part of the core definition of done.  
**Reason:** The portfolio value is expected to be higher from fair comparison, interpretation, reproducible error analysis, and a polished full-stack laboratory than from prematurely multiplying dataset/task complexity.  
**Impact:** Core Definition of Done no longer requires GoEmotions or multi-label support.

## Decision: Separate README from the internal project plan

**Date:** 2026-08-20  
**Status:** accepted  
**Context:** `README.md` and `PROJECT_PLAN.md` were duplicates, which would cause documentation drift and make the public project entry point too planning-heavy.  
**Decision:** `README.md` is the concise public-facing project entry point; `PROJECT_PLAN.md` is the detailed internal product/engineering specification.  
**Reason:** The files now have different audiences and responsibilities.  
**Impact:** Future detailed planning changes belong in `PROJECT_PLAN.md`; README should summarize implemented capabilities and actual findings, not mirror the plan.

## Decision: Use the DAIR.AI Emotion `split` configuration for the core benchmark

**Date:** 2026-08-20  
**Status:** accepted  
**Context:** P2 requires a stable single-label benchmark with fixed held-out examples so all three model families can be compared fairly.  
**Decision:** Use Hugging Face `dair-ai/emotion`, configuration `split`, with the upstream train/validation/test boundaries and verified six-label order. Pin the dataset repository revision in the dataset config for reproducibility.  
**Alternatives considered:** Use the 416,809-row `unsplit` configuration and create new local splits; switch immediately to GoEmotions.  
**Reason:** The provided 16k/2k/2k split is simple, reproducible, and directly supports the project's same-held-out-example comparison requirement.  
**Impact:** `ml/configs/dair_ai_emotion.yaml`, dataset inspection/preparation, all initial model experiments, and later comparison/error artifacts must preserve these upstream split boundaries and label IDs.
## Decision: Use max sequence length 64 for DAIR.AI Emotion

**Date:** 2026-08-20
**Status:** accepted
**Context:** Training-set token-length analysis showed p95=41, p99=52, max=66, and only 14 of 16,000 examples above 60 tokens.
**Decision:** Use `max_sequence_length: 64` for the first DAIR.AI Emotion neural-model pipeline.
**Alternatives considered:** 80 tokens from the original planning draft; 60 tokens; exact maximum of 66.
**Reason:** 64 covers essentially the full observed distribution while avoiding unnecessary padding and retaining a convenient fixed sequence length.
**Impact:** NumPy sequence encoding, TextCNN batching, inference preprocessing, smoke tests, and tokenizer metadata should use length 64.
## Decision: Promote NumPy MLP v002

**Date:** 2026-08-21
**Status:** accepted
**Context:** The initial NumPy MLP v001 trained for only 15 epochs and underfit badly. A controlled v002 experiment changed only the training duration from 15 to 50 epochs.
**Decision:** Promote `numpy_mlp_dair_v002` as the educational NumPy model.
**Reason:** v002 achieved test accuracy 0.8770, macro F1 0.8058, and weighted F1 0.8742, outperforming the TF-IDF Logistic Regression baseline.
**Impact:** v002 becomes the NumPy reference model for later Compare Lab evaluation.