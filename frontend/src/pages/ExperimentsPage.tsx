import { useState } from "react";
import { Link } from "react-router";
import {
  HeldoutSection,
  SourceLink,
  TableScroll,
  ValidationPlot,
} from "../components/ResearchFigures";
import { HistoricalEvidence } from "../components/HistoricalEvidence";
import {
  runs,
  strategies,
  rareLabels,
  strategyNames,
  metricNames,
  statistics,
  fmt,
  research,
  type Metric,
  type Strategy,
  type RareLabel,
} from "../data/research";
export function ExperimentsPage() {
  const [metric, setMetric] = useState<Metric>("macro_f1");
  const [aggregate, setAggregate] = useState(false);
  const [selected, setSelected] = useState<Strategy[]>([...strategies]);
  const [rare, setRare] = useState<RareLabel>("nervousness");
  const filtered = runs.filter((run) => selected.includes(run.strategy));
  function toggle(strategy: Strategy) {
    setSelected((current) =>
      current.includes(strategy)
        ? current.filter((s) => s !== strategy)
        : strategies.filter((s) => s === strategy || current.includes(s)),
    );
  }
  return (
    <div className="page explorer-page">
      <header className="page-heading">
        <p className="eyebrow">Study P11 / Experiment explorer</p>
        <h1>
          Twelve runs.
          <br />
          <span>A closer look.</span>
        </h1>
        <p className="intro">
          Follow the variation across seeds. Compare sampling strategies. Keep
          the evidence in view.
        </p>
        <Link to="/research">Read the study</Link>
      </header>
      <nav className="section-links" aria-label="Explorer sections">
        <a href="#validation">Validation</a>
        <a href="#rare-labels">Rare labels</a>
        <a href="#heldout">Held-out test</a>
        <a href="#historical">Historical evidence</a>
      </nav>
      <p className="provenance-note">
        GoEmotions simplified / 28-label multi-label / validation experiments
      </p>
      <section id="validation" className="research-section" tabIndex={-1}>
        <div className="section-heading">
          <div>
            <p className="eyebrow">01 / Validation</p>
            <h2>Same budget. Different exposure.</h2>
          </div>
          <span className="caption">5,426 examples · 3 paired seeds</span>
        </div>
        <div className="explorer-controls">
          <label>
            Metric
            <select
              aria-label="Metric"
              value={metric}
              onChange={(e) => setMetric(e.target.value as Metric)}
            >
              {Object.entries(metricNames).map(([value, name]) => (
                <option value={value} key={value}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <fieldset className="segmented">
            <legend>View</legend>
            <label>
              <input
                type="radio"
                name="view"
                checked={!aggregate}
                onChange={() => setAggregate(false)}
              />
              Per seed
            </label>
            <label>
              <input
                type="radio"
                name="view"
                checked={aggregate}
                onChange={() => setAggregate(true)}
              />
              Mean ± SD
            </label>
          </fieldset>
          <fieldset className="strategy-filters">
            <legend>Sampling strategy</legend>
            {strategies.map((strategy) => (
              <label key={strategy}>
                <input
                  type="checkbox"
                  checked={selected.includes(strategy)}
                  onChange={() => toggle(strategy)}
                />
                {strategyNames[strategy]}
              </label>
            ))}
          </fieldset>
          <button
            className="secondary-button"
            onClick={() => setSelected(["rare4x", "hybrid"])}
          >
            Compare Real 4× &amp; Hybrid
          </button>
        </div>
        <p className="selection-status" role="status">
          {filtered.length} of 12 validation runs shown · {metricNames[metric]}{" "}
          · {aggregate ? "mean and sample SD" : "per seed"}
        </p>
        {filtered.length ? (
          <>
            <ValidationPlot
              metric={metric}
              selected={selected}
              aggregate={aggregate}
            />
            <TableScroll label="Validation run data">
              <table className="run-table">
                <caption>
                  {metricNames[metric]} ·{" "}
                  {aggregate
                    ? "mean ± sample SD (n = 3)"
                    : "per-seed validation results"}
                </caption>
                <thead>
                  <tr>
                    <th scope="col">Strategy</th>
                    <th scope="col">{aggregate ? "Runs" : "Seed"}</th>
                    <th scope="col">{metricNames[metric]}</th>
                    <th scope="col">{aggregate ? "Sample SD" : "Threshold"}</th>
                    {!aggregate && <th scope="col">Run details</th>}
                  </tr>
                </thead>
                <tbody>
                  {aggregate
                    ? selected.map((strategy) => {
                        const stats = statistics(
                          runs
                            .filter((r) => r.strategy === strategy)
                            .map((r) => r.metrics[metric]),
                        );
                        return (
                          <tr key={strategy}>
                            <th scope="row">{strategyNames[strategy]}</th>
                            <td>3</td>
                            <td>{fmt(stats.mean, 6)}</td>
                            <td>{fmt(stats.sd, 6)}</td>
                          </tr>
                        );
                      })
                    : filtered.map((run) => (
                        <tr key={run.id}>
                          <th scope="row">{strategyNames[run.strategy]}</th>
                          <td>{run.seed}</td>
                          <td>{fmt(run.metrics[metric], 6)}</td>
                          <td>{fmt(run.threshold, 2)}</td>
                          <td>
                            <details>
                              <summary>
                                Inspect {run.seed} /{" "}
                                {strategyNames[run.strategy]}
                              </summary>
                              <div className="run-detail">
                                <p>
                                  <strong>{run.id}</strong>
                                </p>
                                <dl>
                                  <dt>Checkpoint</dt>
                                  <dd>{run.checkpoint}</dd>
                                  <dt>Optimizer steps</dt>
                                  <dd>
                                    {run.optimizerSteps.toLocaleString("en-US")}
                                  </dd>
                                  <dt>Source model</dt>
                                  <dd>{run.provenance.model_id}</dd>
                                  <dt>Pinned revision</dt>
                                  <dd>
                                    <code>{run.provenance.model_revision}</code>
                                  </dd>
                                  <dt>Base config SHA-256</dt>
                                  <dd>
                                    <code>
                                      {run.provenance.base_config_sha256}
                                    </code>
                                  </dd>
                                </dl>
                                {Object.entries(run.metrics).map(
                                  ([key, value]) => (
                                    <p key={key}>
                                      {metricNames[key as Metric]}: {value}
                                    </p>
                                  ),
                                )}
                                {run.sources.map((path) => (
                                  <p key={path}>
                                    <SourceLink name={path} />
                                  </p>
                                ))}
                              </div>
                            </details>
                          </td>
                        </tr>
                      ))}
                </tbody>
              </table>
            </TableScroll>
          </>
        ) : (
          <p className="notice">Select a strategy to inspect its runs.</p>
        )}
        <p className="figure-note">
          Checkpoint selection maximized validation macro AP; thresholds were
          selected by validation macro F1. Charts and tables use the same source
          values. Sample SD uses n − 1. Three seeds do not establish
          significance; small differences are descriptive.{" "}
          <SourceLink name="multiseed_robustness_v001.json">
            Validation source
          </SourceLink>
        </p>
      </section>
      <section id="rare-labels" className="research-section" tabIndex={-1}>
        <p className="eyebrow">02 / Rare-label comparisons</p>
        <h2>Look beyond the average.</h2>
        <label className="rare-selector">
          Emotion
          <select
            aria-label="Emotion"
            value={rare}
            onChange={(e) => setRare(e.target.value as RareLabel)}
          >
            {rareLabels.map((label) => (
              <option key={label}>{label}</option>
            ))}
          </select>
        </label>
        <p>
          Validation support: <strong>{research.rare.support[rare]}</strong>{" "}
          positive examples. Small supports make individual-label estimates
          uncertain. The strategy filters above also apply here.
        </p>
        <TableScroll label="Rare-label per-seed results">
          <table>
            <caption>{rare} · validation per seed</caption>
            <thead>
              <tr>
                <th scope="col">Strategy</th>
                <th scope="col">Seed</th>
                <th scope="col">F1</th>
                <th scope="col">AP</th>
                <th scope="col">Support</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((run) => (
                <tr key={run.id}>
                  <th scope="row">{strategyNames[run.strategy]}</th>
                  <td>{run.seed}</td>
                  <td>{fmt(run.perLabel[rare].f1, 6)}</td>
                  <td>
                    {run.perLabel[rare].average_precision === undefined
                      ? "Not recorded"
                      : fmt(run.perLabel[rare].average_precision!, 6)}
                  </td>
                  <td>{run.perLabel[rare].support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </TableScroll>
        <TableScroll label="Paired hybrid minus real oversampling differences">
          <table>
            <caption>{rare} · paired F1 difference: Hybrid − Real 4×</caption>
            <thead>
              <tr>
                <th scope="col">Seed</th>
                <th scope="col">Real 4× F1</th>
                <th scope="col">Hybrid F1</th>
                <th scope="col">Difference</th>
              </tr>
            </thead>
            <tbody>
              {[42, 123, 456].map((seed) => {
                const real = runs.find(
                  (r) => r.strategy === "rare4x" && r.seed === seed,
                )!.perLabel[rare].f1;
                const hybrid = runs.find(
                  (r) => r.strategy === "hybrid" && r.seed === seed,
                )!.perLabel[rare].f1;
                return (
                  <tr key={seed}>
                    <th scope="row">{seed}</th>
                    <td>{fmt(real, 6)}</td>
                    <td>{fmt(hybrid, 6)}</td>
                    <td>
                      {hybrid - real >= 0 ? "+" : ""}
                      {fmt(hybrid - real, 6)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </TableScroll>
        <p className="figure-note">
          The paired comparison always displays Real 4× and Hybrid, independent
          of the visibility filters. Hybrid improved nervousness in all three
          seeds; other rare labels were mixed. This is descriptive, not a
          significance claim.{" "}
          <SourceLink name="multiseed_rare_label_analysis_v001.json">
            Rare-label evidence
          </SourceLink>
        </p>
      </section>
      <HeldoutSection />
      <HistoricalEvidence />
    </div>
  );
}
