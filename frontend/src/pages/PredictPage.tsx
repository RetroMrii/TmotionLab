import {
  type FormEvent,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import { Link } from "react-router";
import {
  analyzeText,
  getHealth,
  getModels,
  TmotionApiError,
} from "../api/client";
import { EmotionSignal } from "../components/EmotionSignal";
import { TidalDotField } from "../components/TidalDotField";
import { InterfaceIcon } from "../components/InterfaceIcon";
import { emotionFamilies, emotionFamily } from "../emotionColors";
import { keyboardInput, motionTokens } from "../motion";
import type { AnalyzeResponse, LabelScore } from "../types/api";

const PRESETS = [
  [
    "A small joy",
    "I’m so glad we finally made time for this. It means more than you know.",
  ],
  [
    "An uncertain moment",
    "I keep thinking about what might go wrong tomorrow, even though I’ve prepared.",
  ],
  [
    "Mixed feelings",
    "I’ll miss this place and the people here, but I’m excited for what comes next.",
  ],
];
const percent = (score: number) => `${(score * 100).toFixed(1)}%`;
function presentationUntil(deadline: number, signal: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal.aborted) { reject(signal.reason); return; }
    const cancel = () => {
      clearTimeout(timer);
      signal.removeEventListener("abort", cancel);
      reject(signal.reason);
    };
    const timer = setTimeout(() => {
      signal.removeEventListener("abort", cancel);
      resolve();
    }, Math.max(0, deadline - performance.now()));
    signal.addEventListener("abort", cancel, { once: true });
  });
}
function validScores(value: unknown): LabelScore[] {
  if (!Array.isArray(value)) throw new Error("Invalid score response");
  const seen = new Set<string>();
  return value.filter((item): item is LabelScore => {
    if (
      !item ||
      typeof item.label !== "string" ||
      !item.label ||
      !Number.isInteger(item.label_id) ||
      typeof item.score !== "number" ||
      !Number.isFinite(item.score) ||
      item.score < 0 ||
      item.score > 1 ||
      seen.has(item.label)
    )
      return false;
    seen.add(item.label);
    return true;
  });
}
function validateResponse(response: AnalyzeResponse) {
  if (
    response.task !== "multi_label" ||
    !response.input ||
    typeof response.input.text !== "string" ||
    !Array.isArray(response.input.tokens) ||
    !response.input.tokens.every((token) => typeof token === "string") ||
    !Number.isInteger(response.input.token_count) ||
    !Number.isInteger(response.input.original_token_count) ||
    typeof response.model_id !== "string" ||
    typeof response.dataset !== "string" ||
    !Array.isArray(response.detected_labels) ||
    !response.detected_labels.every((label) => typeof label === "string") ||
    !Number.isFinite(response.threshold) ||
    response.threshold < 0 ||
    response.threshold > 1 ||
    typeof response.score_note !== "string" ||
    !Array.isArray(response.availability_notes) ||
    !response.availability_notes.every((note) => typeof note === "string")
  )
    throw new Error("Invalid analysis response");
  return {
    ...response,
    scores: validScores(response.scores),
    ranked_scores: [...validScores(response.ranked_scores)].sort(
      (a, b) => b.score - a.score,
    ),
  };
}
let entered = false;
export function PredictPage() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [availability, setAvailability] = useState(
    "Checking the reading service…",
  );
  const [keyboard, setKeyboard] = useState(false);
  const requestId = useRef(0);
  const controller = useRef<AbortController | null>(null);
  const textarea = useRef<HTMLTextAreaElement>(null);
  const reading = useRef<HTMLElement>(null);
  const submitButton = useRef<HTMLButtonElement>(null);
  const page = useRef<HTMLDivElement>(null);
  useLayoutEffect(() => {
    if (entered || keyboardInput()) return;
    const animations: Animation[] = [];
    const frame = requestAnimationFrame(() => {
      entered = true;
      page.current
        ?.querySelectorAll(".analyze-heading, .workspace")
        .forEach((element, i) =>
          animations.push(
            element.animate(
              [
                { opacity: 0, transform: "translateY(8px)" },
                { opacity: 1, transform: "translateY(0)" },
              ],
              {
                duration: i ? motionTokens.workspace : motionTokens.heading,
                delay: i ? 60 : 0,
                easing: motionTokens.easeOut,
                fill: "backwards",
              },
            ),
          ),
        );
    });
    return () => {
      cancelAnimationFrame(frame);
      animations.forEach((animation) => animation.cancel());
    };
  }, []);
  useEffect(() => {
    let active = true;
    Promise.all([getModels(), getHealth()])
      .then(([registry, health]) => {
        if (!active) return;
        const model = registry.models.find(
          (item) => item.model_id === "roberta_goemotions_rare4x_v001",
        );
        setAvailability(
          model?.runtime_loaded &&
            model.capabilities.includes("live_predict") &&
            health.goemotions_loaded
            ? "Reading service available · GoEmotions · 28 labels"
            : "Reading service unavailable · You can retry when the service recovers.",
        );
      })
      .catch(() => {
        if (active)
          setAvailability(
            "Reading service status unavailable · Try Analyze to reconnect.",
          );
      });
    const ids = requestId,
      pending = controller;
    return () => {
      active = false;
      ids.current++;
      pending.current?.abort();
    };
  }, []);
  useLayoutEffect(() => {
    if (!result || !reading.current) return;
    const animations =
      !keyboard
        ? [...reading.current.querySelectorAll("[data-reveal]")].map(
            (node, i) =>
              node.animate(
                [
                  { opacity: 0, transform: "translateY(8px)" },
                  { opacity: 1, transform: "translateY(0)" },
                ],
                {
                  duration: motionTokens.result,
                  delay: motionTokens.resultDelays[Math.min(i, 4)],
                  easing: motionTokens.easeOut,
                  fill: "backwards",
                },
              ),
          )
        : [];
    const frame = requestAnimationFrame(() => {
      const section = reading.current;
      if (!section) return;
      // Focus never scrolls independently. Don't steal focus if the user moved away while waiting.
      if (keyboard && document.activeElement === submitButton.current) section.focus({ preventScroll: true });
      const anchors = [...section.querySelectorAll(".leading-reading h3, .leading-score strong, .empty-ranking")];
      const rects = anchors.map(node => node.getBoundingClientRect());
      if (!rects.length) return;
      const top = Math.min(...rects.map(rect => rect.top));
      const bottom = Math.max(...rects.map(rect => rect.bottom));
      const viewport = window.visualViewport;
      const visibleTop = (viewport?.offsetTop ?? 0) + 24;
      const visibleBottom = (viewport?.offsetTop ?? 0) + (viewport?.height ?? window.innerHeight) - 24;
      const delta = top < visibleTop || bottom - top > visibleBottom - visibleTop
        ? top - visibleTop : Math.max(0, bottom - visibleBottom);
      if (Math.abs(delta) > 1) window.scrollBy({ top: delta, left: 0, behavior: "smooth" });
    });
    return () => {
      cancelAnimationFrame(frame);
      animations.forEach((a) => a.cancel());
    };
  }, [result, keyboard]);
  function changeText(value: string) {
    requestId.current++;
    controller.current?.abort();
    setText(value);
    setResult(null);
    setError(null);
    setLoading(false);
  }
  async function submit(event: FormEvent) {
    event.preventDefault();
    if (loading || !text.trim()) return;
    controller.current?.abort();
    const current = ++requestId.current;
    const abort = new AbortController();
    controller.current = abort;
    const revealDeadline = performance.now() + motionTokens.analysisPresentation;
    setKeyboard(keyboardInput());
    setResult(null);
    setLoading(true);
    setError(null);
    try {
      // Start inference now; settle both success and failure before presentation, without adding 1.5s after inference.
      const prediction = analyzeText(text, abort.signal).then(validateResponse).then(
        value => ({ ok: true as const, value }),
        cause => ({ ok: false as const, cause }),
      );
      const [outcome] = await Promise.all([prediction, presentationUntil(revealDeadline, abort.signal)]);
      if (current !== requestId.current || abort.signal.aborted) return;
      if (!outcome.ok) throw outcome.cause;
      setResult(outcome.value);
      setAvailability("Last reading completed · GoEmotions · 28 labels");
    } catch (caught) {
      if (current !== requestId.current) return;
      setAvailability("Last reading request failed · You can retry.");
      setError(
        caught instanceof TmotionApiError && caught.status === 503
          ? "The reading service is temporarily unavailable."
          : caught instanceof TypeError
            ? "We couldn’t reach the reading service. Check your connection and try again."
            : "The service could not return a usable reading. Please try again.",
      );
    } finally {
      if (current === requestId.current) setLoading(false);
    }
  }
  const ranked = result?.ranked_scores.slice(0, 3) ?? [];
  const leading = ranked[0];
  return (
    <div className="page analyze-page" ref={page}>
      <TidalDotField analyzing={loading} />
      <section className="analyze-heading">
        <p className="eyebrow">A different perspective</p>
        <h1>
          There’s more
          <br />
          <span>in your words.</span>
        </h1>
        <p className="intro">
          Explore emotion in text. A sentence, a passage, a thought.
          <br className="desktop-break" /> See how the model reads it.
        </p>
      </section>
      <div className="workspace">
        <form className="editor" onSubmit={submit} aria-busy={loading}>
          <div className="editor-rail">
            <label htmlFor="prediction-text">Your text</label>
            <span>English language</span>
          </div>
          <textarea
            ref={textarea}
            id="prediction-text"
            aria-label="English text"
            placeholder="A few words can hold a lot. Start here…"
            maxLength={5000}
            aria-describedby="input-limits"
            value={text}
            onChange={(event) => changeText(event.target.value)}
          />
          <div className="editor-actions">
            <span id="input-limits">
              {text.length.toLocaleString("en-US")} / 5,000
            </span>
            <button
              type="button"
              className="clear-button"
              onClick={() => {
                changeText("");
                textarea.current?.focus();
              }}
            >
              Clear
            </button>
            <button
              ref={submitButton}
              type="submit"
              className="primary-button"
              disabled={!text.trim()}
              aria-disabled={loading}
            >
              {loading ? "Reading your text…" : "Analyze text"}
            </button>
          </div>
        </form>
        <EmotionSignal
          state={
            loading ? "pending" : result ? "resolved" : error ? "error" : "idle"
          }
        />
      </div>
      <div className="presets">
        <span>Or try a thought</span>
        {PRESETS.map(([name, value]) => (
          <button
            key={name}
            onClick={() => {
              changeText(value);
              textarea.current?.focus();
            }}
          >
            {name}
          </button>
        ))}
      </div>
      <p className="sr-only" role="status">
        {loading
          ? "Reading your text. Processing."
          : result
            ? `Reading complete. ${result.detected_labels.length} detected labels. ${result.ranked_scores.length} ranked scores returned.`
            : "Ready for your text."}
      </p>
      {error && (
        <div className="request-error" role="alert">
          <h2>We couldn’t complete that request.</h2>
          <p>{error} Your text is preserved; you can retry.</p>
        </div>
      )}
      {result && (
        <section
          ref={reading}
          className="reading"
          tabIndex={-1}
          aria-label="Your reading"
        >
          <div>
            <div className="reading-heading" data-reveal>
              <div>
                <p className="eyebrow">The model’s perspective</p>
                <h2>Your reading.</h2>
              </div>
              <p>Ranked, not diagnosed.</p>
            </div>
            {leading ? (
              <div className="leading-reading">
                <div>
                  <span className="caption">Highest-scoring emotion</span>
                  <h3 data-emotion-family={emotionFamily(leading.label)}>{leading.label}</h3>
                  <span className="detection-tag">
                    {result.detected_labels.includes(leading.label)
                      ? "Detected"
                      : "Ranked alternative"}
                  </span>
                </div>
                <div className="leading-score">
                  <strong>{percent(leading.score)}</strong>
                  <span>Model probability</span>
                  <small>Uncalibrated score</small>
                </div>
              </div>
            ) : (
              <p className="empty-ranking">
                No valid ranked scores were returned. No leading emotion can be
                shown.
              </p>
            )}
            {ranked.length > 1 && (
              <div className="alternatives">
                <p className="caption">Also in the reading</p>
                {ranked.slice(1).map((item) => (
                  <div className="alternative" key={item.label} data-reveal>
                    <strong data-emotion-family={emotionFamily(item.label)}>{item.label}</strong>
                    <span>
                      {result.detected_labels.includes(item.label)
                        ? "Detected"
                        : "Ranked alternative"}
                    </span>
                    <b>{percent(item.score)}</b>
                  </div>
                ))}
              </div>
            )}
            <div className="detections">
              <span className="caption">
                Detected at threshold {result.threshold.toFixed(2)}
              </span>
              {result.detected_labels.length ? (
                <p className="detected-labels">
                  {result.detected_labels.map((label, i) => <span key={label}>
                    {i > 0 && <span aria-hidden="true"> · </span>}
                    <span data-emotion-family={emotionFamily(label)}>{label}</span>{" "}
                  </span>)}
                </p>
              ) : (
                <p>
                  No labels reached the threshold. Ranking alone does not mean
                  detection.
                </p>
              )}
              <p className="muted">
                Labels are independent and can coexist, including neutral.
                Scores do not sum to 100%.
              </p>
            </div>
          </div>
          <details className="reading-details" data-reveal>
            <summary>
              <span>
                Behind the reading
                <small>All scores, model &amp; analyzed text</small>
              </span>
              <span className="disclosure-symbol" aria-hidden="true">
                <InterfaceIcon name="plus" className="disclosure-closed" />
                <InterfaceIcon name="minus" className="disclosure-open" />
              </span>
            </summary>
            <div className="details-body">
              <h3>All {result.scores.length} valid label scores</h3>
              <p>
                Canonical response order. Bars show uncalibrated sigmoid scores.
              </p>
              <div className="all-scores">
                {result.scores.map((item) => (
                  <div className="score-row" key={item.label} data-emotion-family={emotionFamily(item.label)}>
                    <span>{item.label}</span>
                    <span className="score-track">
                      <span style={{ width: `${item.score * 100}%` }} />
                    </span>
                    <b>{percent(item.score)}</b>
                  </div>
                ))}
              </div>
              <details className="color-guide">
                <summary>About label colors</summary>
                <p>Colors group labels for visual scanning; they do not change predictions or represent a validated psychological taxonomy.</p>
                <dl>{Object.entries(emotionFamilies).map(([family, group]) => <div key={family}>
                  <dt data-emotion-family={family}>{group.name}</dt><dd>{group.labels.join(", ")}</dd>
                </div>)}</dl>
              </details>
              <p>{result.score_note}</p>
              <dl className="metadata">
                <dt>Model</dt>
                <dd>{result.model_id}</dd>
                <dt>Task &amp; dataset</dt>
                <dd>
                  {result.task} · {result.dataset} · seed {result.seed}
                </dd>
                <dt>Tokens read</dt>
                <dd>
                  {result.input.token_count} of{" "}
                  {result.input.original_token_count}, including special tokens
                  · maximum {result.input.max_tokens}
                </dd>
              </dl>
              {result.input.truncated && (
                <p className="notice" role="note">
                  Input was truncated to {result.input.max_tokens} tokens. Only
                  the beginning of this passage was analyzed.
                </p>
              )}
              <h3>Submitted text</h3>
              <blockquote className="exact-text">
                {result.input.text}
              </blockquote>
              <h3>Analyzed tokens</h3>
              <p className="exact-text token-text">
                {result.input.tokens.join(" · ")}
              </p>
              {result.availability_notes.map((note) => (
                <p className="muted" key={note}>
                  {note}
                </p>
              ))}
              <Link to="/research">Read the study behind this model</Link>
            </div>
          </details>
        </section>
      )}
      <section className="context-note">
        <div>
          <p className="eyebrow">A little context</p>
          <h2>A reading, with room for nuance.</h2>
          <p>
            Model scores offer a perspective on language. Human context still
            matters.
          </p>
          <p className="signal-disclaimer">The signal is decorative artwork, not model activity, emotional intensity or an explanation of a prediction.</p>
          <div className="inline-links">
            <Link to="/about">About the project</Link>
            <Link to="/research">Explore the research</Link>
            <Link to="/compare" aria-describedby="historical-context">Compare historical benchmarks</Link>
          </div>
          <p id="historical-context">Historical comparison uses the DAIR six-emotion benchmark, separate from the current GoEmotions model.</p>
        </div>
        <p className="service-status">
          {availability}
          <br />
          <span>Up to 48 tokens · English text</span>
        </p>
      </section>
    </div>
  );
}
