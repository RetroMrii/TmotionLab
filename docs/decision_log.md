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


## Decision: Replace the active DAIR ML track with
## independent GoEmotions multi-label development

**Date:** 2026-10-09
**Status:** accepted

**Context:** The original DAIR single-label laboratory was
completed through P10. The next research objective is
fine-grained multi-label emotion modeling and synthetic
training-data experimentation.

**Decision:**
- Archive the previous ML directory in DAIR-MODEL-OLD.
- Develop the new ML system independently.
- Use the full GoEmotions simplified label taxonomy.
- Begin with a multi-label classical baseline and RoBERTa.
- Follow with controlled synthetic-data research.
- Preserve existing backend, frontend, and test files.
- Defer compatibility and product integration until
  new model experiments are complete.
- Keep the historical project state in the old model archive.
- Continue the shared decision log.

**Reason:** The new research task has different target
semantics, model requirements, evaluation methods, and
data-processing requirements. Backward compatibility
during initial development is unnecessary.

**Impact:** Existing backend imports and tests may fail
temporarily. The original DAIR work remains archived.
New models receive independent configs and artifacts.

## Decision: P12 versioned GoEmotions serving with offline DAIR compatibility

**Date:** 2026-11-04
**Status:** implemented and locally verified
**Context:** P11 is complete; the frozen rare4x seed-42 model exists. DAIR ML code
was archived and old imports prevent the former API from starting, while offline
DAIR benchmark artifacts remain valid. Older P11 status and API 0.3.0 statements
are historical and superseded for integration by this decision.

**Decision:** API 0.4.0 introduces `POST /v1/goemotions/predict` with a version-1,
28-label multi-label schema. Preserve `/predict` and `/interpret` as historical
contracts with explicit 503 responses while their runtime is absent. Load the
frozen local model once per lifespan, verify its files and manifest SHA-256,
require canonical labels, use 48 tokens and fixed sigmoid threshold >= 0.25.
CPU is the default. No test-time tuning, downloads, or fabricated predictions.

Read offline DAIR comparisons, errors and metrics from the archive without
importing or editing archived modules. Registry capabilities reflect working
subsystems. GoEmotions and offline artifact failures are isolated. `/` and
`/predict` in React are Analyze; advanced routes remain discoverable and identify
the six-label historical task. RoBERTa has no TextCNN interpretation capability.

**Alternatives considered:** Breaking `/predict` migration; importing archived
ML into the new package; loading all historical models. The separate versioned
route protects old response semantics and avoids unnecessary archive coupling.

**Validation:** 52 backend tests, real direct inference and 16-row validation
compatibility, live API smoke, browser integration tests, lint/build, artifact
hash comparison and engineering review. See the P12 review for final results.

**Impact:** The frozen 0.3.0 contract remains historical, not a restriction on new
routes. DAIR and GoEmotions metrics are never treated as the same benchmark.
Playwright is a development-only dependency for frontend integration testing.
Full Reading Studio art direction and portable environment pinning remain follow-ups.

## Decision: approved Reading Studio UX refinement — 2026-11-04

**Authority:** The supervisor explicitly authorized implementation of the frontend investigation proposal, with final overrides. Those overrides supersede the proposal's static idle artwork, automatic reduced-motion suppression and explicit View reading recommendation. The earlier approval-gate document remains a historical record.

**Navigation and icon decision:** Historical Compare has one contextual entry in Analyze, labeled “Compare historical benchmarks,” with DAIR provenance. Remove its Explore/About/legacy recovery/historical evidence entry links; retain saved IDs as text, `/compare`, query prefill and backend support. Explore supports mouse hover, a safe pointer corridor, click/touch toggling, Enter/Space, focus retention, Escape and outside/navigation dismissal. Remove unnecessary Unicode arrows and use limited functional SVGs. Source links explicitly say Download.

**Motion decision:** Prefer full motion in the application; do not change browser or OS settings. Remove automatic reduced-motion media-query suppression and the corresponding JavaScript preference listener/checks. Existing unrelated keyboard interactions remain instant. Preserve the original 61-path artwork; continuous 10-second idle motion accelerates through a 650 ms rate ramp to 3.5× during analyzing and decelerates to idle. No score/label geometry coupling, progress percentages or neural-computation claim. Keep the disclaimer visible in Analyze's context section. Offscreen/hidden artwork pauses without affecting inference.

**Accessibility tradeoff:** System reduced-motion preferences no longer automatically protect users from the application's spatial/ambient effects. The approved scope did not include an in-app reduction/pause control. This is an explicit supervisor choice, not an accessibility improvement. Keyboard focus, disclosure semantics, contrast, status/error announcements and activation guards remain required. Automated axe results do not certify motion accessibility.

**Timing and scrolling decision:** Start inference immediately. Reveal actual success/error only after both inference completion and submission + 1.5 seconds; never add 1.5 seconds after a slow inference. Cancel both network and presentation waits on invalidation. Automatically expose only the leading label/number with the smallest necessary native smooth scroll and 24 px visual-viewport insets; already-visible results do not scroll. Keyboard focus uses preventScroll, and only returns to results when the initiating submit retains focus. A focusable aria-disabled busy submit preserves focus while guarding duplicate activation. No View reading button or reserved action row is added. Native smoothing remains browser-dependent; do not force a custom scroll loop.

**Color decision:** Use exactly the approved nine visual families and 28-label dark/light foreground mapping. Family colors apply to names/bars; actual scores, canonical order, threshold 0.25, authoritative detection membership and neutral co-occurrence remain unchanged. The grouping is editorial presentation, not a scientific taxonomy. Existing model/strategy/state colors elsewhere retain their meaning.

**Scope:** Frontend components/styles/tests and engineering/design documentation only. No backend/model/dataset/archive/dependency edits, commit, push or PR. See `reports/engineering/FRONTEND-UX-REFINEMENT-IMPLEMENTATION-REVIEW.md` for final verification and limitations; supervisor acceptance remains pending.

## Decision: Emotion Signal traveling-wave deformation — 2026-11-04

**Authority:** The supervisor's subsequent Vanta Waves reference explicitly requests evolving ribbon curvature and rejects whole-illustration rotation/scaling/fading as the main motion. This supersedes the previous transform-only signal implementation, while retaining the full-motion preference policy, immediate inference, 1.5-second presentation minimum, minimal native result scroll and all other UX refinements.

**Implementation:** Keep the 61 original ribbons, base path geometry, crossing transform and sage/warm gradients. Precompute each point's spatial wave coefficients; animate `d` using four shared temporal coefficients. Fixed endpoints and smoothly tapered outer-ribbon amplitude preserve the composition. Idle principal phase is 10 seconds; analyzing targets 3.5x speed and 1.75x amplitude. A critically damped response preserves phase and transition velocity when interrupted. Hidden/offscreen/unmounted artwork stops scheduling geometry updates. No React state changes per frame and no added dependency.

**Tradeoff:** Nonrigid SVG geometry repaints and performs SVG layout; the old zero-layout transform profile no longer describes this implementation. Local Chrome profiling supports a capped 60-update/s SVG approach. WebGL would add context/shader/dependency and fallback complexity for a small 61-line artwork; it is unnecessary for the measured local target. Physical mobile hardware/battery performance remains unverified. The previously approved reduced-motion accessibility limitation remains explicit. See `reports/engineering/EMOTION-SIGNAL-WAVES-REVIEW.md` for measurements, regression checks and visual evidence.
