import { useId, type ReactNode } from "react";
import {
  research,
  runs,
  strategies,
  rareLabels,
  strategyNames,
  metricNames,
  statistics,
  fmt,
  type Metric,
  type Strategy,
} from "../data/research";
export function SourceLink({
  name,
  children,
}: {
  name: string;
  children?: ReactNode;
}) {
  const source = research.sources.find(
    (s) => s.path === name || s.path.endsWith(`/${name}`),
  );
  return source ? (
    <a
      className="source-link"
      href={source.href}
      download
      title={`SHA-256 ${source.sha256}`}
    >
      Download {children || name.split("/").at(-1)} 
    </a>
  ) : (
    <span>Source unavailable</span>
  );
}
export function TableScroll({
  children,
  label,
}: {
  children: ReactNode;
  label: string;
}) {
  return (
    <div className="table-scroll" tabIndex={0} role="region" aria-label={label}>
      {children}
    </div>
  );
}
export function ValidationPlot({
  metric = "macro_f1",
  selected = strategies,
  aggregate = false,
}: {
  metric?: Metric;
  selected?: Strategy[];
  aggregate?: boolean;
}) {
  const id = useId();
  return (
    <figure className="research-figure">
      <div className="figure-top">
        <span>
          {aggregate ? "Mean ± sample SD" : "Individual training seeds"}
        </span>
        <span>Validation · {metricNames[metric]}</span>
      </div>
      <svg
        className="validation-plot"
        viewBox="0 0 800 290"
        role="img"
        aria-labelledby={id}
      >
        <title id={id}>
          {metricNames[metric]} by strategy on validation.{" "}
          {aggregate
            ? "Mean and sample standard deviation across three seeds."
            : "Seeds 42, 123, 456 shown individually."}{" "}
          Exact values follow in the table.
        </title>
        {[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7].map((tick) => (
          <g key={tick}>
            <line
              x1={160 + (tick / 0.7) * 600}
              x2={160 + (tick / 0.7) * 600}
              y1="30"
              y2="240"
              className="chart-grid"
            />
            <text x={160 + (tick / 0.7) * 600} y="270" textAnchor="middle">
              {tick.toFixed(1)}
            </text>
          </g>
        ))}
        {selected.map((strategy, i) => {
          const values = runs.filter((r) => r.strategy === strategy);
          const stat = statistics(values.map((r) => r.metrics[metric]));
          const y = 55 + i * 56;
          return (
            <g key={strategy} className={`strategy-${strategy}`}>
              <text x="0" y={y + 5}>
                {strategyNames[strategy]}
              </text>
              {aggregate ? (
                <>
                  <line
                    x1={160 + ((stat.mean - stat.sd) / 0.7) * 600}
                    x2={160 + ((stat.mean + stat.sd) / 0.7) * 600}
                    y1={y}
                    y2={y}
                    className="chart-range"
                  />
                  <circle
                    cx={160 + (stat.mean / 0.7) * 600}
                    cy={y}
                    r="3"
                    className="chart-dot"
                  />
                </>
              ) : (
                values.map((run, index) => (
                  <g key={run.seed}>
                    <circle
                      className="chart-dot"
                      cx={160 + (run.metrics[metric] / 0.7) * 600}
                      cy={y + (index - 1) * 12}
                      r={4 + index}
                    />
                    <title>
                      Seed {run.seed}: {fmt(run.metrics[metric], 6)}
                    </title>
                  </g>
                ))
              )}
            </g>
          );
        })}
      </svg>
      <figcaption>
        {aggregate
          ? "Intervals show sample standard deviation, not confidence intervals."
          : "Each point is one run; vertical position separates seeds 42, 123 and 456 within each strategy."}{" "}
        Metric scale 0–1; displayed axis 0–0.7. Three seeds describe variation;
        they do not establish statistical significance.{" "}
        <SourceLink name="multiseed_robustness_v001.json">Source</SourceLink>
      </figcaption>
    </figure>
  );
}
export function SummaryTable() {
  return (
    <TableScroll label="Validation summary table">
      <table>
        <caption>Validation performance · mean ± sample SD, n = 3</caption>
        <thead>
          <tr>
            <th scope="col">Strategy</th>
            {Object.values(metricNames).map((name) => (
              <th scope="col" key={name}>
                {name}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {strategies.map((strategy) => (
            <tr key={strategy}>
              <th scope="row">{strategyNames[strategy]}</th>
              {(Object.keys(metricNames) as Metric[]).map((metric) => {
                const stats = statistics(
                  runs
                    .filter((r) => r.strategy === strategy)
                    .map((r) => r.metrics[metric]),
                );
                return (
                  <td key={metric}>
                    {fmt(stats.mean)} ± {fmt(stats.sd)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </TableScroll>
  );
}
export function SeedTable() {
  return (
    <details>
      <summary>Inspect all twelve validation runs</summary>
      <TableScroll label="All twelve validation runs">
        <table>
          <caption>
            Original seed results · validation · six decimals for display
          </caption>
          <thead>
            <tr>
              <th scope="col">Strategy</th>
              <th scope="col">Seed</th>
              <th scope="col">Macro F1</th>
              <th scope="col">Micro F1</th>
              <th scope="col">Macro AP</th>
              <th scope="col">Threshold</th>
            </tr>
          </thead>
          <tbody>
            {runs.map((run) => (
              <tr key={run.id}>
                <th scope="row">
                  <SourceLink name={run.sources[0]}>
                    {strategyNames[run.strategy]}
                  </SourceLink>
                </th>
                <td>{run.seed}</td>
                <td>{fmt(run.metrics.macro_f1, 6)}</td>
                <td>{fmt(run.metrics.micro_f1, 6)}</td>
                <td>{fmt(run.metrics.macro_average_precision, 6)}</td>
                <td>{fmt(run.threshold, 2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableScroll>
    </details>
  );
}
export function RareFigure() {
  return (
    <figure className="research-figure">
      <div className="figure-top">
        <span>Four rare emotions</span>
        <span>Mean validation F1 · n = 3</span>
      </div>
      <div className="rare-small-multiples">
        {rareLabels.map((label) => (
          <div className="rare-mini" key={label}>
            <h3>{label}</h3>
            <p>Support {research.rare.support[label]}</p>
            {strategies.map((strategy) => {
              const mean = statistics(
                runs
                  .filter((r) => r.strategy === strategy)
                  .map((r) => r.perLabel[label].f1),
              ).mean;
              return (
                <div className="rare-bar-row" key={strategy}>
                  <span>{strategyNames[strategy]}</span>
                  <span className="rare-track">
                    <span
                      className={`strategy-${strategy}`}
                      style={{ width: `${mean * 100}%` }}
                    />
                  </span>
                  <b>{fmt(mean)}</b>
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <figcaption>
        Mean F1, scale 0–1. Numbers are the accessible data alternative.
        Supports count positive validation examples and are shared across runs.{" "}
        <SourceLink name="multiseed_rare_label_analysis_v001.json">
          Rare-label source
        </SourceLink>
      </figcaption>
    </figure>
  );
}
export function HeldoutSection() {
  const test = research.heldout;
  return (
    <section
      id="heldout"
      className="research-section heldout-section"
      tabIndex={-1}
    >
      <p className="eyebrow">Frozen model · held-out test</p>
      <h2>One final evaluation.</h2>
      <p>
        Only the selected Real 4× model, seed 42, was evaluated on the official
        test set: {test.examples.toLocaleString("en-US")} examples, threshold{" "}
        {test.threshold}. The seed was fixed as the reference; it was not
        selected by searching for the best seed. The model and threshold were
        frozen before evaluation, with no test-time tuning.
      </p>
      <p className="selected-model-id">
        <code>{test.experiment_id}</code>
      </p>
      <TableScroll label="Selected model held-out metrics">
        <table>
          <caption>
            Selected model test metrics · no other strategy was evaluated here
          </caption>
          <thead>
            <tr>
              <th scope="col">Metric</th>
              <th scope="col">Score</th>
            </tr>
          </thead>
          <tbody>
            {[
              ["Macro F1", test.macro_f1],
              ["Micro F1", test.micro_f1],
              ["Macro AP", test.macro_average_precision],
              ["Exact-match accuracy", test.exact_match_accuracy],
              ["Example-based F1", test.example_based_f1],
            ].map(([name, value]) => (
              <tr key={name}>
                <th scope="row">{name}</th>
                <td>{fmt(Number(value), 6)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableScroll>
      <TableScroll label="Rare-label held-out metrics">
        <table>
          <caption>
            Rare labels on test · small supports warrant caution
          </caption>
          <thead>
            <tr>
              <th scope="col">Label</th>
              <th scope="col">F1</th>
              <th scope="col">Support</th>
            </tr>
          </thead>
          <tbody>
            {rareLabels.map((label) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td>{fmt(test.per_label[label].f1, 6)}</td>
                <td>{test.per_label[label].support}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </TableScroll>
      <p className="figure-note">
        These test results cannot rank the four training strategies.{" "}
        <SourceLink name="final_heldout_test_v001.json">
          Test metrics
        </SourceLink>{" "}
        ·{" "}
        <SourceLink name="final_model_freeze_v001.json">
          Frozen selection
        </SourceLink>
      </p>
    </section>
  );
}
