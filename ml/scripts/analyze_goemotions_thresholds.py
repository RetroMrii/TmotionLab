import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = (
    ROOT
    / "ml/artifacts/predictions"
    / "roberta_goemotions_rare4x_v001_validation.jsonl"
)
METRICS = (
    ROOT / "ml/artifacts/metrics" / "roberta_goemotions_rare4x_v001_validation.json"
)

BASELINE = 0.25
GRID = np.round(np.arange(0.05, 0.751, 0.01), 2)
MIN_POSITIVES = 20

rows = [
    json.loads(line)
    for line in PREDICTIONS.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

report = json.loads(METRICS.read_text(encoding="utf-8"))
labels = list(report["per_label"])
n_labels = len(labels)

assert len(rows) == 5426, "Unexpected validation row count"
assert n_labels == 28, "Expected 28 emotion labels"

scores = np.asarray([row["probabilities"] for row in rows], dtype=np.float64)
truth = np.zeros((len(rows), n_labels), dtype=np.int8)

assert scores.shape == truth.shape
assert np.isfinite(scores).all()
assert ((scores >= 0) & (scores <= 1)).all()

for i, row in enumerate(rows):
    for label_id in row["true_labels"]:
        truth[i, label_id] = 1

    expected = np.flatnonzero(scores[i] >= BASELINE).tolist()
    assert sorted(row["predicted_labels"]) == expected, (
        f"Prediction mismatch at {row['example_id']}"
    )


def metrics(y, p):
    return {
        "macro": f1_score(y, p, average="macro", zero_division=0),
        "micro": f1_score(y, p, average="micro", zero_division=0),
    }


def fit_thresholds(y, p):
    thresholds = np.full(n_labels, BASELINE, dtype=float)

    for j in range(n_labels):
        if y[:, j].sum() < MIN_POSITIVES:
            continue

        target = y[:, j]
        predictions = p[:, j, None] >= GRID[None, :]

        tp = (predictions & (target[:, None] == 1)).sum(axis=0)
        fp = (predictions & (target[:, None] == 0)).sum(axis=0)
        fn = ((~predictions) & (target[:, None] == 1)).sum(axis=0)

        denom = 2 * tp + fp + fn
        f1 = np.divide(
            2 * tp,
            denom,
            out=np.zeros_like(tp, dtype=float),
            where=denom > 0,
        )

        best = np.flatnonzero(np.isclose(f1, f1.max()))
        closest = best[np.argmin(np.abs(GRID[best] - BASELINE))]
        thresholds[j] = GRID[closest]

    return thresholds


baseline_predictions = scores >= BASELINE
baseline_metrics = metrics(truth, baseline_predictions)

assert abs(baseline_metrics["macro"] - report["macro_f1"]) < 1e-10
assert abs(baseline_metrics["micro"] - report["micro_f1"]) < 1e-10

# Each row receives thresholds fitted without that row.
oof_predictions = np.zeros_like(truth)
fold_thresholds = []

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, eval_idx in kf.split(scores):
    thresholds = fit_thresholds(truth[train_idx], scores[train_idx])
    oof_predictions[eval_idx] = scores[eval_idx] >= thresholds[None, :]
    fold_thresholds.append(thresholds)

oof_metrics = metrics(truth, oof_predictions)

# Descriptive only: these use the entire validation set.
# They are NOT independent evaluation results.
final_thresholds = fit_thresholds(truth, scores)
in_sample_metrics = metrics(truth, scores >= final_thresholds[None, :])

print("\nVALIDATION THRESHOLD DIAGNOSTICS")
print(f"Rows: {len(rows)} | Labels: {n_labels}")
print(
    f"Global 0.25: macro={baseline_metrics['macro']:.4f}, "
    f"micro={baseline_metrics['micro']:.4f}"
)
print(
    f"Out-of-fold: macro={oof_metrics['macro']:.4f}, micro={oof_metrics['micro']:.4f}"
)
print(
    f"Difference:  macro="
    f"{oof_metrics['macro'] - baseline_metrics['macro']:+.4f}, "
    f"micro="
    f"{oof_metrics['micro'] - baseline_metrics['micro']:+.4f}"
)
print(
    f"In-sample:  macro={in_sample_metrics['macro']:.4f} "
    "(optimistic; not validation evidence)"
)

print("\nPER-LABEL RESULTS")
print(
    f"{'Label':18} {'Support':>7} {'Base F1':>8} "
    f"{'OOF F1':>8} {'Full thr':>9} {'Fold range':>11}"
)

for j, name in enumerate(labels):
    base = f1_score(truth[:, j], baseline_predictions[:, j], zero_division=0)
    oof = f1_score(truth[:, j], oof_predictions[:, j], zero_division=0)
    selected = np.array([t[j] for t in fold_thresholds])

    print(
        f"{name:18} {truth[:, j].sum():7d} "
        f"{base:8.4f} {oof:8.4f} "
        f"{final_thresholds[j]:9.2f} "
        f"{selected.min():.2f}-{selected.max():.2f}"
    )

print("\nNo production settings or model files changed.")
