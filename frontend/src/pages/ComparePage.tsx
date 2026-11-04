import { useEffect, useRef, useState, type FormEvent } from "react";
import { Link } from "react-router";
import { getComparison, getModels } from "../api/client";
import type { CompareResponse } from "../types/api";
const IDS = [
  "dair_test_001837",
  "dair_test_001943",
  "dair_test_000353",
  "dair_test_000265",
  "dair_test_000576",
];
export function ComparePage() {
  const [id, setId] = useState(
    () => new URLSearchParams(window.location.search).get("example") || IDS[0],
  );
  const [result, setResult] = useState<CompareResponse | null>(null);
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("Checking historical artifacts…");
  const generation = useRef(0),
    abort = useRef<AbortController | null>(null);
  useEffect(() => {
    let active = true;
    getModels()
      .then(({ models }) => {
        if (active)
          setStatus(
            models.some((m) => m.capabilities.includes("held_out_compare"))
              ? "Saved historical comparisons available"
              : "Historical comparison unavailable",
          );
      })
      .catch(() => {
        if (active) setStatus("Historical service status unavailable");
      });
    const ids = generation,
      pending = abort;
    return () => {
      active = false;
      ids.current++;
      pending.current?.abort();
    };
  }, []);
  function change(value: string) {
    generation.current++;
    abort.current?.abort();
    setId(value);
    setResult(null);
    setError("");
    setPending(false);
  }
  async function load(value: string) {
    abort.current?.abort();
    const current = ++generation.current;
    const controller = new AbortController();
    abort.current = controller;
    setResult(null);
    setError("");
    setId(value);
    setPending(true);
    try {
      const response = await getComparison(value.trim(), controller.signal);
      if (current === generation.current) {
        setResult(response);
        setStatus("Saved comparison loaded");
      }
    } catch {
      if (current === generation.current)
        setError(
          "We couldn’t load that saved example. Check the ID or try again when the historical service is available.",
        );
    } finally {
      if (current === generation.current) setPending(false);
    }
  }
  function submit(event: FormEvent) {
    event.preventDefault();
    if (id.trim() && !pending) void load(id);
  }
  const rows = result
    ? ([
        [
          "TF-IDF + Logistic Regression",
          "Historical baseline",
          result.baseline_prediction,
          result.baseline_score,
          result.baseline_correct,
        ],
        [
          "NumPy MLP",
          "Historical educational model",
          result.numpy_prediction,
          result.numpy_score,
          result.numpy_correct,
        ],
        [
          "TextCNN",
          "Historical neural model",
          result.textcnn_prediction,
          result.textcnn_score,
          result.textcnn_correct,
        ],
      ] as const)
    : [];
  return (
    <div className="page compare-page">
      <header className="page-heading">
        <p className="eyebrow">Compare perspectives</p>
        <h1>
          One passage.
          <br />
          <span>Different perspectives.</span>
        </h1>
        <p className="intro">
          Three model families reading the same saved passage. Look for
          agreement, difference, and the limits of a single label.
        </p>
      </header>
      <p className="provenance-note">
        Historical DAIR.AI · six-label, single-label benchmark. This comparison
        is separate from the live GoEmotions model.
      </p>
      <section className="comparison-input">
        <form onSubmit={submit}>
          <label htmlFor="example-id">Saved example ID</label>
          <div className="input-action">
            <input
              id="example-id"
              value={id}
              onChange={(event) => change(event.target.value)}
              spellCheck={false}
            />
            <button className="primary-button" disabled={!id.trim() || pending}>
              {pending ? "Loading…" : "Load comparison"}{" "}

            </button>
          </div>
        </form>
        <div className="presets">
          <span>Explore a saved case</span>
          {IDS.map((value, i) => (
            <button
              key={value}
              onClick={() => void load(value)}
              aria-label={`Load ${value}`}
            >
              0{i + 1}

            </button>
          ))}
        </div>
        <p className="muted" role="status">
          {pending ? "Loading saved comparison…" : status}
        </p>
      </section>
      {error && (
        <div role="alert" className="request-error">
          <h2>Comparison unavailable</h2>
          <p>{error}</p>
        </div>
      )}
      {!result && !error && (
        <section className="compare-empty">
          <span className="folio">01 — 03</span>
          <h2>A shared starting point.</h2>
          <p>
            Load a saved test example to place each model’s prediction beside
            the benchmark label. This instrument uses existing example IDs.
          </p>
        </section>
      )}
      {result && (
        <section className="comparison-result">
          <div className="comparison-caption">
            <span>{result.example_id}</span>
            <span>
              Benchmark label: <strong>{result.true_label_name}</strong>
            </span>
          </div>
          <blockquote className="comparison-quote">“{result.text}”</blockquote>
          <div
            className="comparison-table"
            role="table"
            aria-label="Historical model predictions"
          >
            <div className="comparison-row comparison-table-head" role="row">
              <span role="columnheader">Model</span>
              <span role="columnheader">Prediction</span>
              <span role="columnheader">Raw score</span>
              <span role="columnheader">Benchmark</span>
            </div>
            {rows.map(([name, role, label, score, correct]) => (
              <div className="comparison-row" role="row" key={name}>
                <div role="cell">
                  <strong>{name}</strong>
                  <small>{role}</small>
                </div>
                <span role="cell" className="comparison-label">
                  {label}
                </span>
                <span role="cell" className="numeric">
                  {Number.isFinite(score) ? score.toFixed(4) : "Unavailable"}
                  {Number.isFinite(score) && score >= 0 && score <= 1 && (
                    <span className="compare-score-track" aria-hidden="true">
                      <span style={{ width: `${score * 100}%` }} />
                    </span>
                  )}
                </span>
                <span role="cell" className={correct ? "match" : "difference"}>
                  {correct ? "Matches" : "Differs"}
                </span>
              </div>
            ))}
          </div>
          <p className="figure-note">
            Raw scores are uncalibrated. Different model families do not share a
            comparable probability scale.
          </p>
          <div className="agreement-summary">
            <h2>Where the readings meet.</h2>
            <dl>
              <dt>All models agree</dt>
              <dd>{result.all_models_agree ? "Yes" : "No"}</dd>
              <dt>All match the benchmark</dt>
              <dd>{result.all_models_correct ? "Yes" : "No"}</dd>
              <dt>All differ from the benchmark</dt>
              <dd>{result.all_models_wrong ? "Yes" : "No"}</dd>
              <dt>Three different predictions</dt>
              <dd>{result.three_way_disagreement ? "Yes" : "No"}</dd>
            </dl>
          </div>
          <div className="case-explanation">
            {result.all_agree_and_wrong && (
              <p>
                All three models choose the same alternative. Agreement does not
                make that label a benchmark match, or settle the meaning of the
                passage.
              </p>
            )}
            {result.baseline_correct_textcnn_wrong && (
              <p>
                The baseline matches the benchmark while TextCNN differs: a
                historical TextCNN regression on this example.
              </p>
            )}
            {result.baseline_wrong_textcnn_correct && (
              <p>
                TextCNN matches the benchmark where the baseline differs: a
                historical correction on this example.
              </p>
            )}
            {result.three_way_disagreement && (
              <p>
                Each model chooses a different label. Their disagreement makes
                the effect of the learned representation visible.
              </p>
            )}
          </div>
          <details>
            <summary>Inspect agreement flags</summary>
            <dl className="metadata">
              {Object.entries(result)
                .filter(([, value]) => typeof value === "boolean")
                .map(([key, value]) => (
                  <div className="metadata-pair" key={key}>
                    <dt>{key.replaceAll("_", " ")}</dt>
                    <dd>{value ? "Yes" : "No"}</dd>
                  </div>
                ))}
            </dl>
          </details>
        </section>
      )}
      <div className="context-note">
        <p>
          Benchmark disagreement is an evaluation result, not a complete account
          of a passage’s meaning.
        </p>
        <Link to="/research/experiments#historical">
          Inspect historical evidence
        </Link>
      </div>
    </div>
  );
}
