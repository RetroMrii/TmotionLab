# Reading Studio & Research Edition — validation

**Latest motion follow-up:** the Vanta Waves reference request now uses actual
ribbon curvature deformation. Its [wave-motion review](../reports/engineering/EMOTION-SIGNAL-WAVES-REVIEW.md)
supersedes the transform-only signal measurements below; other UX behavior is
retained. Earlier validation records remain historical evidence.

Wave follow-up checks: **64 tests passed (3.5m)**, lint/build passed, 12 axe
scans reported zero violations, and 209 protected files are unchanged. Forty-four
state screenshots and desktop/mobile-width profiles confirm evolving geometry
near 60 updates/s, with higher rendering cost than the former transform loop.
The request/result-scroll source, navigation shell, emotion map and dependency
files are unchanged. Physical-mobile performance remains unverified.

**Current refinement validation (2026-11-04):** see the appended approved UX
refinement section and the [implementation review](../reports/engineering/FRONTEND-UX-REFINEMENT-IMPLEMENTATION-REVIEW.md).
The original redesign record below is historical: its reduced-motion policy,
request-only artwork and result focus/scroll description were superseded by the
supervisor's explicit implementation overrides.

Completed locally on 2026-11-04 in `C:\TmotionLab`. This records fresh checks for
the post-P12 redesign, separately from the earlier P12 integration review.

## Delivered experience

Analyze (`/` and `/predict`), historical Compare, the Research paper, Experiment
Explorer, About and compatibility/fallback views now form one frontend. The
editor leads; an explicitly artistic optical signal responds to request state;
the reading continues below the workspace with an immediate leading label,
actual score, alternatives, authoritative detections and full disclosure.

The design uses the requested slate/sage dark and light palettes, self-hosted
Geist, thin rules, restrained corners and a lens mark shared with the favicon.
One active `src/index.css` replaces the previous global styling; unused
`App.css` was removed. The Research paper has a section index, numbered figures,
methodology, caveats, source downloads and a separate frozen-test evaluation.
The Explorer exposes all twelve runs, metric/mode/strategy controls, sample SD,
per-run provenance and rare-label paired differences. Compare retains its
historical saved-example workflow with aligned model rows and real agreement
flags. About uses an editorial chapter structure.

See [the preimplementation art direction](reading_studio_art_direction.md).

## Audit and boundaries

- The checkout already contained uncommitted P11 and P12 work. It was preserved;
  no commit, push, PR, retraining, model cleanup or backend migration was done.
- Fresh audit found React, Router and Vite, but no Geist or Motion dependency.
  Geist was added. CSS and WAAPI meet the motion requirements without an
  animation dependency; native SVG/CSS provide the research figures.
- The requested skills were not initially present. The original
  [frontend-design](../.agents/skills/frontend-design/SKILL.md),
  [emil-design-eng](../.agents/skills/emil-design-eng/SKILL.md),
  [animate](../.agents/skills/animate/SKILL.md) and
  [review-animations](../.agents/skills/review-animations/SKILL.md) skills were
  installed under `.agents/skills` and read, including motion recipes/standards.
  Sources: Anthropic's `skills` repository and Emil Kowalski's `skills` repository.
  The installed browser tooling and Playwright performed rendered verification.
- The live API advertised GoEmotions readiness and historical artifact
  availability. Live Analyze and saved Compare were tested against that API.
- A fresh before/after SHA-256 inventory covers **209 protected files**, including
  backend code/tests, selected model artifacts, dataset manifests/data, P11
  evidence and the historical archive. **No protected file changed.** Inventories:
  `reports/engineering/reading-studio-integrity-before.json` and `-after.json`.

## Main implementation files

| Area | Files |
|---|---|
| Routing and shell | `frontend/src/App.tsx`, `components/AppShell.tsx` |
| Analyze and lens | `pages/PredictPage.tsx`, `components/EmotionSignal.tsx`, `motion.ts` |
| Historical comparison | `pages/ComparePage.tsx`, optional AbortSignal in `api/client.ts` |
| Research | `pages/ResearchPage.tsx`, `pages/ExperimentsPage.tsx`, `components/ResearchFigures.tsx`, `components/HistoricalEvidence.tsx` |
| Editorial/supporting views | `pages/AboutPage.tsx`, `pages/NotFoundPage.tsx` |
| Theme and typography | `src/index.css`, early theme script in `index.html`, `public/favicon.svg` |
| Evidence export | `scripts/export-research.mjs`, `research-sources.lock.json`, `src/data/research.json`, `src/data/research.ts`, `public/research/sources/` |
| QA | `tests/analyze.spec.ts`, `tests/setup.ts`, `playwright.config.ts` |

## Preserved contracts and route compatibility

The backend and API schemas were not edited. Analyze still uses
`POST /v1/goemotions/predict`, independent sigmoid scores, 28 labels, threshold
0.25, 5,000 characters and 48 tokens including special tokens. It preserves the
exact submitted text, reports truncation, sorts a copy of valid ranked scores,
does not pad empty/fewer rankings and uses the returned detection set. Neutral
can coexist with other labels; zero detections are a valid state. Malformed
responses fail honestly. Abort controllers plus generation guards prevent old
responses from reappearing after editing, clearing, presets, navigation or a
failed replacement.

Frontend `/errors` and `/metrics` redirect to
`/research/experiments#historical`. This section retains historical DAIR metrics,
the first 20 saved error cases with their actual total count, and the top ten
confusion pairs. Error IDs link to a prefilled Compare ID. Original historical
API endpoints remain intact. `/interpret` explains the unavailable historical
runtime; the current GoEmotions model has no attribution capability or Interpret
CTA. Unknown routes have a designed recovery page. `/predict` marks Analyze as
the active navigation destination.

## Research export and scientific checks

The deterministic build-time exporter reads **38 original source files**. It
copies only compact reports, metrics, configuration, generation prompt,
manifests and provenance. No model weights or individual held-out predictions
are shipped. The generated research module is independent of FastAPI.

The lock manifest contains the complete source paths and SHA-256 hashes. Every
download URL resolves to an exact byte copy under `/research/sources/`. Builds
run `research:check`, which checks both the original hashes and generated output
bytes. Normal builds never regenerate or repin changed evidence. Explicit
`--add-sources` was used to add the inspected annotation audit and generation
prompt while continuing to verify existing pins.

Validation checks cover four strategies × seeds 42/123/456, original metric-file
hashes from the robustness report, validation split identity, no test evaluation
on those runs, thresholds, 5,408 optimizer steps, metric bounds, rare-label F1
and supports, and recomputed means/sample SD against the saved summary. The
selected test report is checked against its freeze hash, model identity,
threshold, seed and no-test-tuning flag. Chart and table data share the same
validated values. Rounding is presentational; original precision is retained in
the data, run details and downloadable metrics.

Key original evidence fingerprints:

| Source | SHA-256 |
|---|---|
| `multiseed_robustness_v001.json` | `e217bf39024b1394ceb736e66f25a9a9f9f4b9f2fd7855f007ea58f087b7cb8c` |
| `multiseed_rare_label_analysis_v001.json` | `c67dd4bb596061e980ab20477270b160b58f70e226d648355ebe6be3d45c82cd` |
| `final_heldout_test_v001.json` | `e640be4a23a77b5755ca56520af5125715053b49dc97eaf7e9227da56916f3df` |
| `final_model_freeze_v001.json` | `01868a7770a1a8f3be4ca75d5b4518b36305376d422b134dbeee1d4224bc86a1` |

Scientific distinctions remain visible: three seeds are descriptive; hybrid's
small mean lead is inconclusive; synthetic labels were not independently
human-verified; supports are small; validation influenced selection and threshold
tuning; official validation/test overlap is a caveat. Only selected Real 4×
seed 42 has a test panel. No test bars or claims were invented for other
strategies. Proposed future work is identified as proposed.

## Executed commands

Commands were run from the repository root unless otherwise noted.

| Command | Fresh result |
|---|---|
| `node frontend/scripts/export-research.mjs --pin` | Exit 0; initial 36-source export |
| `node frontend/scripts/export-research.mjs --add-sources` | Exit 0; 38 sources, all existing pins unchanged |
| `npm --prefix frontend run lint` | Exit 0; no warnings in final run |
| `npm --prefix frontend run build` | Exit 0; export checks, TypeScript and Vite succeed |
| `npm --prefix frontend test -- --reporter=list` | Exit 0; **34 passed** |
| `.venv\Scripts\python.exe -m pytest tests/backend -q -p no:cacheprovider --basetemp .venv/reading-studio-pytest` | Exit 0; **52 passed** |
| `.venv\Scripts\python.exe backend/scripts/smoke_api.py` | Exit 0; live health/models/Analyze/Compare/metrics/errors/confusions, and expected legacy 503 responses |
| Before/after integrity comparison | Exit 0; **209 files unchanged** |
| `git diff --check` | Exit 0; no whitespace errors; normal LF→CRLF notices retained |

Logs are in `reports/engineering/reading-studio-{lint,build,tests-final,backend-tests,api-smoke}.txt`.
The line-ending configuration was not changed. Dependency installation reported
the same two high-severity advisories already recorded during P12; dependency
security remediation was not part of this redesign.

## Browser evidence

Chrome/Playwright executed **48 primary combinations**: six routes (`/`,
`/predict`, `/compare`, `/research`, `/research/experiments`, `/about`) × widths
1440/1024/768/390 × explicit persisted dark/light themes. All passed visible
content, populated-state and no-horizontal-page-overflow checks. **Eight real
Analyze result layouts** and their expanded full-score disclosures passed too.
No page-level JavaScript exceptions occurred in the primary matrix. The in-app
browser was also used to inspect the mobile composition and submit a live preset
reading; its model response and result discovery were visibly verified.

Other executed cases cover blank/preset/clear behavior, truncation, unsorted and
invalid scores, empty/fewer rankings, zero/one/multiple detections including
neutral, malformed responses, real service failure, duplicate pending requests,
all replacement races, Compare replacement errors, all compatibility routes,
theme reload persistence, Escape/focus return, outside-click/focus-leave closing,
keyboard destination/result focus, reduced motion, synchronized explorer
controls, actual source downloads and API-independent research.

Screenshots are in `reports/engineering/reading-studio/`:

- `matrix-*.png`: all 48 primary route layouts, with real populated Compare.
- `result-*.png`, `details-*.png`: eight live readings and disclosures.
- `paper-opening-*.png`, `paper-results-*.png`, `paper-rare-*.png`: focused paper review at 1440/390 in both themes.
- `aggregate-*.png`, `rare-*.png`, `heldout-*.png`: focused Explorer evidence.
- `menu-*.png`, `mobile-menu-*.png`: desktop and mobile Explore in both themes.
- `pending-desktop-dark.png`, `pending-mobile-reduced-*.png`: controlled pending states.
- `real-service-unavailable.png`, `explorer-aggregate-offline.png`, `fallback.png`: failure, offline research and recovery.

### Fixtures and service failure

Successful matrix readings and truncation checks use the actual frozen local
model. Compare uses the real historical artifact API. Every Research figure uses
original P11 data. Browser-only response interception in `tests/analyze.spec.ts`
isolates rare UI cases and delayed race responses; those fixtures are never
imported by production code. Pending screenshots use controlled intercepted
requests, not an imposed production loading delay.

For a real network failure, QA starts a separate Vite service on port 5174 with
API origin `127.0.0.1:65530`, which is unreachable. No response interception is
used in that case. Analyze shows connection failure and preserves its text;
the twelve-run static Explorer remains usable. The user's primary API is never
stopped. The QA server runs in process and closes after tests. An initial Windows
child-process teardown hung after assertions completed; this was fixed by the
in-process setup, and the final suite exits cleanly.

### Findings fixed during verification

- Closed desktop Explore initially extended scroll width at tablet sizes; its
  alignment is now bounded and all four widths pass.
- Native disclosure and nested-select test names were initially too narrow;
  explicit accessible select names and correct summary locators resolved them.
- An initial axe scan sampled partially transparent reveal frames. Scans now
  await finite animations before evaluating the settled reading; no contrast
  rule was disabled and the requested palette was preserved.
- Mobile signal caption and chart labels were enlarged after visual review.
- The per-seed figure now has its own twelve-row tabular equivalent on the paper.

## Accessibility evidence and limits

Twelve fresh axe scans (six populated states × two themes) report **zero
violations** for WCAG 2 A/AA and WCAG 2.1 AA tags. They cover expanded Analyze
Details, open navigation, populated Compare, Research, Explorer with a run
expanded, and About. Machine-readable results are `reading-studio/axe-*.json`.

Axe flagged some SVG text and decorative/overlapping-layout nodes for manual
contrast review rather than reporting violations. These were inspected in
rendered screenshots and against their solid theme tokens. Supplemental contrast
calculations: signal muted text 6.64:1 dark / 4.61:1 light; chart text 7.79:1 /
5.53:1; active navigation accent 9.63:1 / 5.50:1; light flow arrow 4.88:1. Signal
copy occupies a separate area from the visible artwork. Chart values have
adjacent tables/text alternatives, and strategy meaning is not color-only.

Keyboard, focus, native disclosures, reduced motion and contained horizontal
tables were exercised. This is scoped automated and visual evidence, not a full
WCAG conformance certification or a screen-reader/physical-device audit.

## Separate animation review

Reviewed after implementation using
[review-animations](../.agents/skills/review-animations/SKILL.md) and its
[standards](../.agents/skills/review-animations/STANDARDS.md).

| Before | After | Why |
|---|---|---|
| First-entry flag could be consumed by React's development effect cleanup | Entry is scheduled in a cancellable frame, and marked entered only when it runs (`PredictPage.tsx:99`) | Preserves a single first entry under Strict Mode and avoids replay on ordinary navigation |
| Leading label and score shared the delayed reveal | Label renders immediately; only the score enters at the 50 ms step (`PredictPage.tsx:348`) | Users can read the primary result immediately |
| Mobile disclosure's `display:none` made its motion abrupt | Origin-aware starting style and discrete display transition retain 200 ms entry / 120 ms exit (`index.css:1883`) | Preserves mobile spatial continuity while closing controls become inert immediately (`AppShell.tsx:155`) |
| CSS keyboard rules alone did not cancel a WAAPI animation already running | Keyboard input and a newly enabled reduced-motion preference cancel active animations (`AppShell.tsx:53`) | Keyboard actions remain immediate, including interruption during an entrance |

**Verdict: Approve.** No unresolved blocking motion finding remains.

- **Purpose/frequency:** one first-session introduction; occasional result
  composition; request-only state indication; short pointer feedback. Research
  data, rows, numbers and paragraphs remain static.
- **Timing/easing:** shared `.23,1,.32,1` ease-out; `.77,0,.175,1` signal settle;
  100/160 ms press/release, 200/120 ms navigation, 280 ms result steps with
  0/50/100/150/230 ms delays. The 320 ms first workspace and 450 ms artistic
  signal settle are explicit brief exceptions to the ordinary sub-300 ms budget.
- **Performance:** only transform/opacity move; SVG paths are built once at
  module scope and memoized. No per-frame React updates, idle loops, number
  tweening, score-bar sweeps or layout-property animation. Theme color changes
  are a short, bounded paint transition explicitly requested by the brief.
- **Interruption:** menu/signal transitions retarget; request effects cancel on
  cleanup; replacement immediately removes old results; discovery uses one
  cancellable frame. No fading stale controls remain interactive.
- **Accessibility:** pointer-only press effects and gated hover styles; keyboard
  actions have no choreography; reduced motion removes spatial/continuous
  effects and smooth scrolling. Pending-loop start/stop, keyboard/reduced result
  discovery and mobile disclosure were tested and rendered.

File references above are relative to `frontend/src/`. The in-app browser and
screenshots support the visual judgment; no measured frame-rate or physical
device performance claim is made.

## Remaining limits

Live Analyze still depends on FastAPI and the retained local GoEmotions model.
Historical Compare/evidence depends on the archived-artifact API. Historical
live TextCNN inference/interpretation remains unavailable, and RoBERTa exposes
no token attribution. Research exports require the pinned original local source
files at build time; the completed static build does not require inference.
Clean-machine deployment, GPU operation and cross-browser/physical-device
certification remain outside this frontend task.

Verified preview: [Reading Studio](http://127.0.0.1:5173/),
[Research](http://127.0.0.1:5173/research),
[Experiment Explorer](http://127.0.0.1:5173/research/experiments).

## Approved UX refinement — 2026-11-04

The supervisor authorized all five investigated changes, with continuous idle
motion, full-motion preference handling, a minimum 1.5-second presentation
sequence and automatic minimal scrolling. No View reading button or reserved
action row was added. The proposal remains a historical approval record; the
[implementation review](../reports/engineering/FRONTEND-UX-REFINEMENT-IMPLEMENTATION-REVIEW.md)
documents the actual final behavior, file-level scope, all nine color families,
separate motion review and remaining limitations.

Fresh verification: **58 Playwright tests passed**, followed by **2 passing
targeted narrow-layout tests** after the final <=360 px result stacking change.
Lint and the research-check/TypeScript/Vite build pass. The 48-route layout
matrix and eight live Analyze result/detail layouts cover 1440/1024/768/390 in
both themes; additional 320 px fixtures exercise long labels. Twelve populated
axe scans report zero violations, with incomplete/manual items retained. The
evidence directory holds 110 browser screenshots plus raw timing, scroll,
palette and motion logs: [validation summary](../reports/engineering/ux-refinement/validation-summary.json).

The suite covers actual Analyze inference and truncation, live historical
Compare, real connection refusal, research without inference, exact 28-label
colors/order/scores/detection semantics, neutral co-occurrence, pointer/keyboard
result discovery, touch/hover Explore, duplicate activation, and cancellation
both during inference and after completion while presentation is waiting.
Timing/color/race fixtures are controlled test responses, not claimed model
outputs. Static screenshots freeze animation; separate animation assertions
and profiling inspect live motion.

The fast controlled response started its API request at +0.5 ms, completed at
+10.0 ms and revealed at +1519.3 ms. The slow response completed at +1733.2 ms
and revealed at +1760.5 ms: no extra 1.5-second delay. These measurements include
normal scheduling/render latency, not a promise of frame-exact delivery.
Default Chrome still snaps for a native smooth-scroll request; the application
now requests only the minimum label/score reveal distance. The diagnostic
Chrome process with native smoothing enabled traverses intermediate positions.
Already-visible results do not scroll, and keyboard focus uses preventScroll.

The original 61 SVG path strings retain their pre-edit SHA-256. A 10-second
WAAPI wrapper loop ramps to 3.5x over 650 ms during analyzing and back to idle
on completion/error/cancellation. HTML-wrapper animation eliminated measured
steady-state SVG layout work; bounded inner settling still has a rendering
cost. Offscreen pause, visible resume and unmount cleanup were verified. The
scientific disclaimer is visible in Analyze's context section.

**Accessibility tradeoff:** full motion intentionally overrides automatic
system reduced-motion handling, without changing OS/browser settings. No
in-app pause/reduction control is included. Existing unrelated keyboard
interactions remain instant. This is not motion-accessibility or WCAG
certification; physical devices, screen readers, exhaustive zoom/forced-colors
and other browser engines remain outside the verified scope.

All 209 protected backend/model/data/archive/research files are hash-identical;
the build validates 38 pinned research sources. No dependency, API contract,
threshold or model change was made. Earlier backend test counts are historical
and were not rerun for this frontend-only refinement. No commit, push or PR was
created. Implementation awaits supervisor acceptance.
