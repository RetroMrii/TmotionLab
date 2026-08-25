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

## Decision: Keep Error Lab benchmark-relative and defer human adjudication

**Date:** 2026-08-25  
**Status:** accepted  
**Context:** During P8, several high-score model/dataset disagreements were semantically plausible alternatives, and in some cases all three model families selected the same non-benchmark label. Treating every mismatch as an unambiguous semantic model failure would overstate what the dataset supports.  
**Decision:** Error Lab will continue to define an evaluation error reproducibly as `prediction != dataset label`, while user-facing language will prefer **benchmark disagreement** or **evaluation error**. Systematic manual relabeling, annotation adjudication, and reviewer-agreement studies are deferred.  
**Alternatives considered:** Manually review and categorize all 207 TextCNN disagreements; relabel questionable dataset cases; present every mismatch as a model mistake.  
**Reason:** The benchmark-relative definition is reproducible and supports the Error Lab without introducing an incomplete or subjective annotation study.  
**Impact:** Reports, API responses, and frontend copy must state that benchmark disagreement does not prove semantic model failure. Existing review-queue tooling is not required for the current Definition of Done.

## Decision: Freeze the first FastAPI serving contract around TextCNN live inference plus saved analysis artifacts

**Date:** 2026-08-25  
**Status:** accepted  
**Context:** P6-P8 already produce reproducible comparison, interpretation, metrics, and error-analysis artifacts. Loading all three trained models for every frontend feature would add serving complexity and would blur the distinction between live inference and held-out benchmark analysis.  
**Decision:** The P9 backend loads the promoted TextCNN once for live `/predict` and `/interpret` requests, while `/compare`, `/metrics`, `/errors`, and `/errors/confusions` serve previously generated benchmark artifacts. Baseline and NumPy MLP are not loaded into the live serving path. API version `0.3.0` is the frozen contract for the first React frontend implementation.  
**Alternatives considered:** Load all three models for arbitrary live comparison; recompute benchmark analysis per request; let the frontend read artifact files directly.  
**Reason:** The chosen design keeps runtime serving small, preserves the exact held-out comparison contract, and separates live model behavior from reproducible offline evaluation.  
**Impact:** `/compare/{example_id}` is held-out-example based, `/models` reports TextCNN as the only runtime-loaded model, and React consumes stable FastAPI contracts instead of duplicating analysis logic.

## Decision: Build the frontend as an ML laboratory with six explicit views

**Date:** 2026-08-25  
**Status:** accepted  
**Context:** The portfolio thesis requires the interface to demonstrate model behavior rather than look like a generic emotion-classification demo.  
**Decision:** The React frontend is organized into Home, Predict, Compare, Interpret, Error Lab, Metrics, and About/Methodology routes, with the four thesis views—Predict, Compare, Interpret, Error Lab—treated as first-class interactions.  
**Alternatives considered:** Single-page classifier UI; combine all analysis into one dashboard; expose only live prediction.  
**Reason:** Separate views make each experimental question legible and keep the UI aligned with the underlying ML workflow and saved artifacts.  
**Impact:** React routing, shared API types, navigation, responsive behavior, and frontend integration tests/manual gates are part of P10. Compare remains benchmark-example based, Interpret preserves non-causal wording, and Error Lab preserves benchmark-relative semantics.

## Decision: Keep score language conservative across API and frontend

**Date:** 2026-08-25  
**Status:** accepted  
**Context:** TextCNN returns softmax probabilities, but no calibration experiment has been performed. Baseline, NumPy MLP, and TextCNN also produce scores from different model families and decision functions.  
**Decision:** Use **probability** or **raw model score** rather than calibrated confidence. Do not assume raw scores are directly comparable across model families.  
**Alternatives considered:** Label every top score as confidence; normalize scores in the UI without a calibration study.  
**Reason:** The conservative language matches what has actually been measured and avoids implying probabilistic guarantees that the experiments do not establish.  
**Impact:** API schemas, Predict, Compare, Error Lab, Metrics, and About copy must preserve this distinction until calibration is explicitly implemented and evaluated.
