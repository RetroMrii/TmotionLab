import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "ml/artifacts/optimization/opt007"
REF = ROOT / ("ml/artifacts/metrics/roberta_goemotions_rare4x_v001_validation.json")
OUTPUT = ROOT / ("reports/goemotions/optimization/opt009_ap_robustness.json")

SEED = 42
REPLICATES = 300


def load_rows(mode):
    path = RUNS / mode / "validation_predictions.jsonl"
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


reference = json.loads(REF.read_text(encoding="utf-8"))
labels = list(reference["per_label"])

baseline = load_rows("baseline")
weighted = load_rows("weighted")

assert len(baseline) == len(weighted) == 5426
assert len(labels) == 28

ids_a = [r["example_id"] for r in baseline]
ids_b = [r["example_id"] for r in weighted]

assert ids_a == ids_b
assert len(set(ids_a)) == len(ids_a)

n = len(baseline)
k = len(labels)

truth = np.zeros((n, k), dtype=np.int8)

for i, (a, b) in enumerate(zip(baseline, weighted)):
    assert sorted(a["true_labels"]) == sorted(b["true_labels"])
    truth[i, a["true_labels"]] = 1

a_scores = np.asarray(
    [r["probabilities"] for r in baseline],
    dtype=np.float64,
)
b_scores = np.asarray(
    [r["probabilities"] for r in weighted],
    dtype=np.float64,
)

assert a_scores.shape == b_scores.shape == truth.shape
assert np.isfinite(a_scores).all()
assert np.isfinite(b_scores).all()
assert np.all((a_scores >= 0) & (a_scores <= 1))
assert np.all((b_scores >= 0) & (b_scores <= 1))

support = truth.sum(axis=0)

# Support groups are descriptive, not alternative
# pre-registered model-selection criteria.
groups = {
    "all_28": list(range(k)),
    "support_below_30": [j for j in range(k) if support[j] < 30],
    "support_30_to_99": [j for j in range(k) if 30 <= support[j] < 100],
    "support_100_plus": [j for j in range(k) if support[j] >= 100],
    "excluding_grief_relief": [
        j for j, name in enumerate(labels) if name not in {"grief", "relief"}
    ],
}


def label_ap(y, scores):
    # Labels absent from a bootstrap sample are
    # undefined for AP; use NaN and record them.
    values = np.full(k, np.nan, dtype=float)

    for j in range(k):
        positives = y[:, j].sum()

        if positives == 0 or positives == len(y):
            continue

        values[j] = average_precision_score(y[:, j], scores[:, j])

    return values


def summarize_delta(delta, indices):
    selected = delta[indices]
    valid = np.isfinite(selected)

    if not valid.all():
        return None

    return float(selected.mean())


original_a = label_ap(truth, a_scores)
original_b = label_ap(truth, b_scores)
original_diff = original_b - original_a

original = {
    name: summarize_delta(original_diff, indices) for name, indices in groups.items()
}

rng = np.random.default_rng(SEED)
bootstrap = {name: [] for name in groups}
skipped = {name: 0 for name in groups}

for iteration in range(REPLICATES):
    sampled = rng.integers(0, n, size=n)

    y = truth[sampled]
    a = label_ap(y, a_scores[sampled])
    b = label_ap(y, b_scores[sampled])
    difference = b - a

    for name, indices in groups.items():
        value = summarize_delta(difference, indices)
        if value is None:
            skipped[name] += 1
        else:
            bootstrap[name].append(value)


def interval(values):
    if not values:
        return None

    values = np.asarray(values, dtype=float)

    return {
        "lower_2_5_percentile": float(np.quantile(values, 0.025)),
        "median": float(np.median(values)),
        "upper_97_5_percentile": float(np.quantile(values, 0.975)),
        "share_positive": float(np.mean(values > 0)),
        "valid_replicates": len(values),
    }


report = {
    "experiment": "OPT-009",
    "model": "RoBERTa-base",
    "split": "validation",
    "examples": n,
    "bootstrap_replicates": REPLICATES,
    "random_state": SEED,
    "sampling": "paired row bootstrap with replacement",
    "support_groups": {
        name: {
            "labels": [labels[j] for j in indices],
            "original_ap_difference": original[name],
            "bootstrap": interval(bootstrap[name]),
            "skipped_replicates": skipped[name],
        }
        for name, indices in groups.items()
    },
    "per_label": [
        {
            "label": name,
            "support": int(support[j]),
            "baseline_ap": float(original_a[j]),
            "weighted_ap": float(original_b[j]),
            "difference": float(original_diff[j]),
        }
        for j, name in enumerate(labels)
    ],
    "limitations": [
        "Bootstrap intervals are conditional on the fitted checkpoints.",
        "Validation was used previously for checkpoint selection.",
        "No independent test evaluation was performed.",
        "Small rare-label supports produce unstable estimates.",
        "This experiment does not change model settings.",
    ],
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print("\nOPT-009 — AP ROBUSTNESS ANALYSIS")
print(f"Rows: {n}, labels: {k}")
print(f"Bootstrap replicates: {REPLICATES}")

print("\nSUPPORT GROUP COMPARISON")
for name, data in report["support_groups"].items():
    ci = data["bootstrap"]
    if ci is None:
        print(f"{name}: insufficient bootstrap evidence")
        continue

    print(
        f"{name:25} "
        f"labels={len(data['labels']):2} "
        f"delta={data['original_ap_difference']:+.4f} "
        f"interval=[{ci['lower_2_5_percentile']:+.4f}, "
        f"{ci['upper_97_5_percentile']:+.4f}] "
        f"positive={ci['share_positive']:.3f} "
        f"valid={ci['valid_replicates']} "
        f"skipped={data['skipped_replicates']}"
    )

print("\nPER-LABEL AP DIFFERENCES")
for row in sorted(
    report["per_label"],
    key=lambda x: x["difference"],
    reverse=True,
):
    print(
        f"{row['label']:18} support={row['support']:4} delta={row['difference']:+.4f}"
    )

print(f"\nSaved: {OUTPUT}")
print("No training, model, or production changes.")
