# Reading Studio & Research Edition — art direction

## Thesis

A quiet instrument for reading language, paired with an inspectable research publication. Slate, sage and warm mineral tones; generous typography; fine rules instead of a dashboard of cards. The main action is writing. The signature is an optical field of thin overlapping ribbons, explicitly artistic rather than attribution.

## Composition and materials

Desktop: an 80 px horizontal shell, max 1280 px; content max 1200 px with 96 px total gutters (64 at 1100, 48 at 768, 40 at 560). Compact two-line opening above one 62/38 editor/signal workspace. Results continue below a fine rule. On mobile the editor leads and a centered, text-free 156 px signal follows; action becomes full width. No text overlays the artwork. At 360 px and below, stack the leading label/score and alternative rows to keep long emotion names readable without horizontal overflow.

```
mark / name                 Analyze   Explore   About   theme
eyebrow
There's more / in your words.
support
┌ editor, textarea, counter, action ┬ optical signal ┐
└─────────────────────────────────┴────────────────┘
fill-only example passages
────────────────────────────────────────────────────
Your reading.       ranked leading label / raw score
also in the reading / actual detection set / disclosure
context                                           footer
```

Use the brief's exact dark/light tokens in one active index.css. Self-host Geist Variable; regular and medium weights, 15–16 px body, 19/17 px editor, 46–64 px main heading, 64–80 px leading label. Borders and dividers establish structure; 10–12 px corners only on containing surfaces. Controls have visible outlines and 44 px targets. Lens mark is reused by the favicon.

## Interaction and motion

Supervisor-approved UX refinement (2026-11-04): the application explicitly prefers full motion and does not automatically suppress effects for `prefers-reduced-motion`. It does not change OS/browser settings. This intentionally gives up automatic accommodation for motion-sensitive users; no in-app pause/reduction control was approved. Unrelated keyboard-triggered controls retain their instantaneous behavior and existing durations. The ambient ribbon is exempt from blanket keyboard animation cancellation.

First-session entry remains heading 280 ms, workspace 320 ms with 60 ms delay and 8 px travel. The subsequent Vanta Waves motion request replaces whole-artwork rotation/scale/opacity with actual ribbon deformation. Preserve all 61 ribbons, their 85-sample original resting geometry, fixed crossing arrangement and sage/warm strokes. Two traveling waves share four temporal coefficients across the path field; the principal idle phase takes 10 s and a slower secondary phase prevents a mechanical single loop. Analyzing smoothly raises speed to 3.5x and deformation amplitude to 1.75x; a critically damped response settles over approximately 650 ms and preserves velocity through reversal. Resolved/error/cancellation return to idle without restarting phase. Outer ribbon deformation tapers smoothly to avoid clipping. The artwork's global transform is fixed; there is no additional settle/scale/fade effect. A frame loop updates only SVG path data, capped near 60 updates/s, and stops while hidden/offscreen or unmounted. No per-frame React state, filter or WebGL dependency is used. There is no score-derived geometry or inferred model activity. The scientific disclaimer remains in Analyze's context section, away from the artwork. See the [wave-motion review](../reports/engineering/EMOTION-SIGNAL-WAVES-REVIEW.md) for the repaint cost and verification.

Inference starts immediately. Presentation waits until both inference has settled and submission + 1,500 ms; slow inference receives no additional 1,500 ms. Abort/generation guards cover network and presentation waits. The genuine analyzing state continues until reveal. Apply the same minimum to error presentation; cancellation ends it immediately.

Results reveal automatically, with no View reading control or reserved action row. The leading label and numeric score render immediately at reveal. Measure their union against the visual viewport with 24 px comfort insets; scroll only the necessary vertical delta, requesting native smooth scrolling. Do nothing when that union already fits. Keyboard completion focuses the reading with `preventScroll:true` only while the initiating submit still has focus, so focus cannot generate a second scroll or steal focus from a subsequent task. Keep the busy submit focusable but activation-disabled via aria-disabled plus the request guard. Native smooth scrolling may still snap in some Chrome configurations; do not force a custom scroll animator. Existing secondary result entrances retain 280 ms and shared staging tokens. No animated numbers or score-bar sweeps.

Explore retains its existing 200 ms entry / 120 ms exit, trigger origin and inert closing state. Mouse hover opens, a 7 px hit corridor and 150 ms exit grace protect travel, keyboard Enter/Space and touch toggle it, Escape restores focus without hover reopening, and outside focus/interaction/navigation dismiss it. Two research destinations remain. Compare's only inbound UI entry is “Compare historical benchmarks” in Analyze's contextual links. Saved Compare routes/query-prefill and all historical content remain; historical table IDs are selectable text rather than alternate Compare links.

Remove redundant action/internal-link arrows. Use a small local SVG family only for Explore destination arrows, plus/minus disclosure and mobile menu/close. Download links say Download; study-flow and benchmark/prediction direction remain meaningful text. Pointer press remains 100/release160 ms and scale .98. Shared easing remains cubic-bezier(.23,1,.32,1) and (.77,0,.175,1).

Emotion label colors use the approved nine-family, 28-label mapping in `frontend/src/emotionColors.ts`, with exact dark/light foreground tokens in `frontend/src/index.css`. Names and score-bar fills use family colors; percentages and detection words remain neutral. No hue depends on rank, probability or threshold. Canonical order, authoritative detections and neutral co-occurrence stay intact. These are editorial visual groups, not a psychological taxonomy. Behind the reading includes a compact explanatory disclosure.
The Reading Studio dependency audit found no Motion library. CSS transitions and WAAPI provide the specified effects without adding one. Native disclosure semantics and explicit focus handling implement navigation. The earlier redesign added Geist as a runtime dependency and axe for QA; this refinement changes no dependencies. The frontend-design, emil-design-eng, animate and review-animations skills installed under .agents/skills inform the implementation; browser verification uses Playwright and rendered screenshots.

## Scientific and editorial structure

Analyze preserves generation/abort guards, exact input, canonical scores and authoritative detections. Sort valid ranked entries on a copy; never pad absent predictions. Threshold is distinct from ranking and uncalibrated scores are distinct from confidence. No GoEmotions interpretation action. Service availability comes from current API responses.

Compare remains a saved DAIR example instrument, aligned by model with explicit benchmark match flags and raw scores. The Research overview reads as a paper: abstract, hypotheses, data, methods, results, rare labels, frozen held-out test, discussion, limitations, conclusion and reproducibility. Numbered captions and adjacent source links identify claims. A flow figure, validation seed plot and rare-label figure have accessible tables. The explorer offers all twelve validation runs, metric/mode/strategy controls, descriptive mean ± sample SD, paired rare-label differences and a separate selected-model test section. Three seeds do not establish significance; unverified synthetic labels and very small supports remain prominent.

## Evidence contract

Generate a small static research export from original metrics, provenance and saved reports. Pin SHA-256 hashes in a manifest; build checks fail if inputs or generated output diverge. Download links serve exact source copies, excluding predictions and weights. Validate 4 strategies × 3 seeds, split identity, thresholds, metric bounds, original report hashes and sample statistics. Research stays readable with the inference API offline. Keep all backend, datasets, model weights and original P11 evidence untouched.

## Verification and review

The approved refinement verifies immutable-file hashes, lint/build, 58 frontend tests and two final narrow-layout checks, live inference/Compare and diff whitespace. Six primary routes at 1440/1024/768/390 in both themes, eight successful Analyze layouts and 320 px long-label fixtures cover responsive composition. Tests cover empty/pending/result, disclosures, desktop/mobile navigation, populated Compare, research controls, fallbacks, network/presentation race cancellation, empty/fewer/invalid rankings, detections, focus, persistence and full motion under either system preference. Twelve populated axe scans report zero violations; motion has a separate measured profile and Before/After/Why review. Backend tests were not rerun in this frontend-only refinement; 209 protected files remain unchanged.

## Current acceptance state

The implemented changes preserve the routes, lens identity, exact palette, mobile hierarchy, scientific provenance and request lifecycle. No backend, dependency or chart redesign was required. Native Chrome smoothing remains configuration-dependent, and the approved full-motion policy has an explicit accessibility cost. See [the implementation review](../reports/engineering/FRONTEND-UX-REFINEMENT-IMPLEMENTATION-REVIEW.md) for screenshots, evidence and the supervisor acceptance gate.
