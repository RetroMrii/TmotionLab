import evidence from "../data/optimization.json";
import { TableScroll } from "./ResearchFigures";

export function OptimizationSummary() {
  const deltas = [...evidence.deltas].sort((a, b) => b.delta_ap - a.delta_ap);
  return <section id="current-model" className="research-section" aria-labelledby="current-model-title">
    <p className="eyebrow">Current model · OPT-010 · validation evidence</p>
    <h2 id="current-model-title">A measured amount of synthetic text.</h2>
    <p>The reading studio now uses RoBERTa with 10% synthetic training draws and real rare-label 4× sampling.
      GoEmotions adds 27 emotions plus neutral and permits overlapping labels, beyond the historical six-label DAIR task.
      Ten percent achieved the highest macro average precision and macro F1 within the four OPT-010 mixtures.
      Micro F1 was slightly lower than the real-only control.</p>
    <p>All OPT-010 runs used standard BCE, seed 42, four epochs, 5,408 optimizer steps and the same 5,426 validation examples.
      Checkpoint selection used validation macro AP. The deployed threshold remains a fixed global 0.25; it is not a per-label tuned policy.</p>
    <TableScroll label="Optimization validation results">
      <table><caption>Saved validation results · threshold 0.25 · one seed per configuration</caption>
        <thead><tr><th scope="col">Study / encoder</th><th scope="col">Configuration</th><th scope="col">Macro AP</th><th scope="col">Macro F1</th><th scope="col">Micro F1</th></tr></thead>
        <tbody>{evidence.runs.map(run => <tr key={`${run.experiment}-${run.mode}`}>
          <th scope="row">{run.experiment.replace("OPT", "OPT-")} · {run.experiment === "OPT012" ? "DeBERTa-v3-base" : "RoBERTa-base"}</th>
          <td>{run.mode}{run.experiment === "OPT010" && run.mode === "synthetic10" ? " · deployed" : ""}</td>
          <td>{run.macro_average_precision.toFixed(6)}</td><td>{run.macro_f1.toFixed(6)}</td><td>{run.micro_f1.toFixed(6)}</td>
        </tr>)}</tbody>
      </table>
    </TableScroll>
    <p>Macro AP measures ranking quality averaged equally across labels, without choosing a classification threshold.
      Macro F1 averages per-label precision/recall balance; micro F1 pools decisions across labels and is influenced more by frequent labels.</p>
    <details><summary>What changed across the optimization studies?</summary>
      <p>OPT-007: positive-class-weighted BCE improved macro AP but reduced thresholded macro and micro F1.
        OPT-008: five-fold out-of-fold thresholds gave standard BCE macro F1 {evidence.opt008.baseline.out_of_fold.macro_f1.toFixed(6)} and micro F1 {evidence.opt008.baseline.out_of_fold.micro_f1.toFixed(6)}.
        Threshold fitting was cross-fitted, but checkpoint selection still used validation.</p>
      <p>OPT-009 generated {evidence.synthetic.source_candidates.toLocaleString("en-US")} candidates; {evidence.synthetic.excluded} unique examples were excluded and {evidence.synthetic.retained.toLocaleString("en-US")} retained.
        Integrity, semantic similarity and annotation-review checks overlap. Human review covered samples, not every retained example; remaining labels may be wrong.</p>
      <p>OPT-011 aligned the two OPT-010 prediction files and inspected every label. The largest gains were concentrated in grief and pride, with only 13 and 15 positive validation examples.
        Out-of-fold threshold optimization did not recover the micro F1 tradeoff.</p>
      <TableScroll label="Synthetic10 per-label AP differences">
        <table><caption>OPT-011 · synthetic10 minus OPT-010 control · all 28 labels</caption>
          <thead><tr><th scope="col">Emotion</th><th scope="col">Validation support</th><th scope="col">AP difference</th></tr></thead>
          <tbody>{deltas.map(row => <tr key={row.label}><th scope="row">{row.label}</th><td>{row.support}</td><td>{row.delta_ap > 0 ? "+" : ""}{row.delta_ap.toFixed(4)}</td></tr>)}</tbody>
        </table>
      </TableScroll>
      <p>OPT-012: an initial DeBERTa FP16 gradient-unscaling error was corrected. Both final runs completed.
        Synthetic10 reached macro AP 0.509283, above its DeBERTa control but below the selected RoBERTa candidate. It was not selected.</p>
    </details>
    <p className="notice">This promotion is based on one seed and reused validation data. Multi-seed replication and bootstrap analysis for this selection were skipped by supervisor decision.
      No independent post-selection test result exists for synthetic10. The historical P11 held-out scores below belong only to rare4x and do not measure the current model.</p>
    <p>Published aggregate evidence is derived from saved metrics and aligned validation predictions; source SHA-256 records are retained in the repository.
      It contains no individual passages or predictions.</p>
  </section>;
}
