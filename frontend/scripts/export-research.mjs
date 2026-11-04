import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import assert from "node:assert/strict";

// No inference, network, predictions or model-weight reads. Sources are immutable.
const root = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const check = process.argv.includes("--check");
const pin = process.argv.includes("--pin");
const extend = process.argv.includes("--add-sources");
const manifestPath = resolve(root, "frontend/research-sources.lock.json");
const pinned = existsSync(manifestPath)
  ? JSON.parse(readFileSync(manifestPath, "utf8"))
  : null;
assert(
  pinned || pin,
  "Initial export requires explicit --pin. Builds never repin sources.",
);
const sources = [];
const outputs = new Map();
const hash = (bytes) => createHash("sha256").update(bytes).digest("hex");
function source(path) {
  const bytes = readFileSync(resolve(root, path));
  const sha256 = hash(bytes);
  if (pinned && (pinned[path] || !extend))
    assert.equal(pinned[path], sha256, `Source changed: ${path}`);
  const href = `/research/sources/${path}`;
  sources.push({ path, sha256, href });
  outputs.set(`frontend/public${href}`, bytes);
  return path.endsWith(".json")
    ? JSON.parse(bytes.toString("utf8"))
    : bytes.toString("utf8");
}
const report = (name) => source(`reports/goemotions/${name}`);
const robustness = report("multiseed_robustness_v001.json");
const rare = report("multiseed_rare_label_analysis_v001.json");
const heldout = report("final_heldout_test_v001.json");
const freeze = report("final_model_freeze_v001.json");
report("goemotions_research_final_v001.md");
const dataset = source("data/processed/goemotions_v001/manifest.json");
source("data/processed/synthetic_goemotions_v002/manifest.json");
source("ml/configs/roberta_goemotions_multilabel_real_v001.json");
report("synthetic_pilot_v002_audit.json");
report("multiseed_sampling_audit_v001.json");
report("controlled_sampling_audit_v001.json");
report("hybrid_sampling_audit_v001.json");
report("annotation_audit_v001.json");
source("ml/configs/synthetic/synthetic_emotion_generation_v002.txt");
const runs = [];
const metrics = ["macro_f1", "micro_f1", "macro_average_precision"];
for (const strategy of ["uniform", "rare4x", "synthetic", "hybrid"]) {
  for (const seed of [42, 123, 456]) {
    const summary = robustness.runs[strategy][seed];
    const id = summary.experiment_id;
    const path = `ml/artifacts/metrics/${id}_validation.json`;
    const metric = source(path);
    assert.equal(sources.at(-1).sha256, summary.metrics_sha256);
    const provenancePath = `ml/artifacts/models/${id}/experiment_provenance.json`;
    const provenance = source(provenancePath);
    assert.equal(metric.evaluation_split, "validation");
    assert.equal(metric.test_evaluated, false);
    assert.equal(provenance.seed, seed);
    assert.equal(metric.optimizer_steps, 5408);
    assert.equal(metric.selected_threshold, summary.selected_threshold);
    for (const key of metrics) {
      assert(
        Number.isFinite(metric[key]) && metric[key] >= 0 && metric[key] <= 1,
      );
      assert.equal(metric[key], summary.metrics[key]);
    }
    for (const label of rare.target_labels) {
      assert.equal(
        metric.per_label[label].f1,
        rare.runs[strategy][seed].per_label[label].f1,
      );
      assert.equal(metric.per_label[label].support, rare.support[label]);
    }
    runs.push({
      strategy,
      seed,
      id,
      threshold: metric.selected_threshold,
      metrics: summary.metrics,
      perLabel: metric.per_label,
      checkpoint: metric.best_checkpoint,
      optimizerSteps: metric.optimizer_steps,
      provenance,
      sources: [path, provenancePath],
    });
  }
  for (const key of metrics) {
    const values = runs
      .filter((r) => r.strategy === strategy)
      .map((r) => r.metrics[key]);
    const mean = values.reduce((a, b) => a + b, 0) / 3;
    const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / 2);
    assert(Math.abs(mean - robustness.summary[strategy][key].mean) < 1e-12);
    assert(Math.abs(sd - robustness.summary[strategy][key].sample_std) < 1e-12);
  }
}
assert.equal(runs.length, 12);
assert.equal(heldout.split, "test");
assert.equal(heldout.experiment_id, freeze.primary_experiment);
assert.equal(heldout.threshold, freeze.classification_threshold);
assert.equal(freeze.seed, 42);
assert.equal(heldout.test_time_tuning_performed, false);
assert.equal(
  heldout.freeze_sha256,
  sources.find((s) => s.path.endsWith("final_model_freeze_v001.json")).sha256,
);
const data = {
  schemaVersion: 1,
  study: "P11",
  runs,
  summary: robustness.summary,
  rare,
  heldout,
  freeze,
  dataset,
  sources,
};
outputs.set(
  "frontend/src/data/research.json",
  Buffer.from(JSON.stringify(data, null, 2) + "\n"),
);
if ((pin && !pinned) || extend)
  writeFileSync(
    manifestPath,
    JSON.stringify(
      Object.fromEntries(sources.map((s) => [s.path, s.sha256])),
      null,
      2,
    ) + "\n",
  );
for (const [path, bytes] of outputs) {
  const full = resolve(root, path);
  if (check)
    assert(readFileSync(full).equals(bytes), `Export differs: ${path}`);
  else {
    mkdirSync(dirname(full), { recursive: true });
    writeFileSync(full, bytes);
  }
}
console.log(
  `${check ? "Verified" : "Exported"} 12 validation runs, 1 selected held-out evaluation and ${sources.length} hash-pinned sources.`,
);
