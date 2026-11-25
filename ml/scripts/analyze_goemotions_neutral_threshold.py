import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = (
    ROOT
    / "ml/artifacts/predictions"
    / "roberta_goemotions_rare4x_v001_validation.jsonl"
)
METRICS = (
    ROOT / "ml/artifacts/metrics" / "roberta_goemotions_rare4x_v001_validation.json"
)
OUTPUT = ROOT / "reports/goemotions/optimization" / "opt004_neutral_threshold.json"

BASELINE = 0.25
GRID = np.round(np.arange(0.05, 0.751, 0.01), 2)

reference = json.loads(METRICS.read_text(encoding="utf-8"))
labels = list(reference["per_label"])
neutral = labels.index("neutral")

with PREDICTIONS.open(encoding="utf-8") as file:
    rows = [json.loads(line) for line in file if line.strip()]

assert len(rows) == 5426
assert len(labels) == 28
assert reference["threshold"] == BASELINE
assert len({r["example_id"] for r in rows}) == len(rows)

scores = np.asarray(
    [r["probabilities"] for r in rows],
    dtype=np.float64,
)

truth = np.zeros((len(rows), len(labels)), dtype=bool)

assert scores.shape == truth.shape
assert np.isfinite(scores).all()
assert ((scores >= 0) & (scores <= 1)).all()

for i, row in enumerate(rows):
    ids = row["true_labels"]
    assert len(ids) == len(set(ids))
    assert all(isinstance(label_id, int) and 0 <= label_id < 28 for label_id in ids)
    truth[i, ids] = True

    expected = np.flatnonzero(scores[i] >= BASELINE).tolist()
    assert sorted(row["predicted_labels"]) == expected


def evaluate(y, pred):
    tn = y[:, neutral]
    pn = pred[:, neutral]

    return {
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, pred, average="micro", zero_division=0)),
        "neutral_f1": float(f1_score(tn, pn, zero_division=0)),
        "neutral_fp": int(np.sum(~tn & pn)),
        "neutral_fn": int(np.sum(tn & ~pn)),
    }


def select_threshold(y, p):
    # Only neutral decisions change; maximizing neutral F1
    # also maximizes macro F1 with all other labels fixed.
    target = y[:, neutral]
    values = p[:, neutral]

    predicted = values[:, None] >= GRID[None, :]

    tp = np.sum(predicted & target[:, None], axis=0)
    fp = np.sum(predicted & ~target[:, None], axis=0)
    fn = np.sum(~predicted & target[:, None], axis=0)

    denom = 2 * tp + fp + fn
    f1_values = np.divide(
        2 * tp,
        denom,
        out=np.zeros_like(tp, dtype=float),
        where=denom > 0,
    )

    best = np.flatnonzero(np.isclose(f1_values, np.max(f1_values)))

    # Prefer the existing threshold when ties occur.
    chosen = best[np.argmin(np.abs(GRID[best] - BASELINE))]
    return float(GRID[chosen])


baseline_pred = scores >= BASELINE
baseline = evaluate(truth, baseline_pred)

assert abs(baseline["macro_f1"] - reference["macro_f1"]) < 1e-10

assert abs(baseline["micro_f1"] - reference["micro_f1"]) < 1e-10

oof_pred = baseline_pred.copy()
folds = []

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

for fold_id, (fit_idx, eval_idx) in enumerate(
    cv.split(scores, truth[:, neutral].astype(int)),
    start=1,
):
    selected = select_threshold(truth[fit_idx], scores[fit_idx])

    oof_pred[eval_idx, neutral] = scores[eval_idx, neutral] >= selected

    fold_baseline = evaluate(truth[eval_idx], baseline_pred[eval_idx])
    fold_candidate = evaluate(truth[eval_idx], oof_pred[eval_idx])

    folds.append(
        {
            "fold": fold_id,
            "threshold": selected,
            "baseline_macro_f1": fold_baseline["macro_f1"],
            "candidate_macro_f1": fold_candidate["macro_f1"],
            "baseline_neutral_f1": fold_baseline["neutral_f1"],
            "candidate_neutral_f1": fold_candidate["neutral_f1"],
        }
    )

oof = evaluate(truth, oof_pred)

# Full-validation fitting is exploratory, not independent.
full_threshold = select_threshold(truth, scores)
full_pred = baseline_pred.copy()
full_pred[:, neutral] = scores[:, neutral] >= full_threshold
in_sample = evaluate(truth, full_pred)

# Verify that non-neutral predictions are identical.
non_neutral = [j for j in range(28) if j != neutral]
assert np.array_equal(
    baseline_pred[:, non_neutral],
    oof_pred[:, non_neutral],
)

result = {
    "experiment": "OPT-004",
    "model": "roberta_goemotions_rare4x_v001",
    "split": "validation",
    "examples": len(rows),
    "baseline_global_threshold": BASELINE,
    "grid": {
        "start": 0.05,
        "stop": 0.75,
        "step": 0.01,
    },
    "method": "5-fold stratified out-of-fold",
    "random_state": 42,
    "baseline": baseline,
    "out_of_fold": oof,
    "difference": {key: oof[key] - baseline[key] for key in baseline},
    "folds": folds,
    "full_validation_fitted_threshold": full_threshold,
    "in_sample_optimistic": in_sample,
    "notes": [
        "Only the neutral threshold was optimized.",
        "All 27 other labels retained threshold 0.25.",
        "Neutral co-occurrence was preserved.",
        "Out-of-fold scores are validation-only estimates.",
        "Full-validation fit is optimistic.",
        "No held-out test data accessed.",
        "No production settings changed.",
    ],
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print("\nOPT-004 — NEUTRAL THRESHOLD OPTIMIZATION")
print(f"Validation examples: {len(rows)}")

print("\nBASELINE vs OUT-OF-FOLD")
for key in baseline:
    print(
        f"{key:18} "
        f"base={baseline[key]:.4f} "
        f"oof={oof[key]:.4f} "
        f"delta={oof[key] - baseline[key]:+.4f}"
    )

print("\nFOLD THRESHOLDS")
for fold in folds:
    print(
        f"Fold {fold['fold']}: "
        f"threshold={fold['threshold']:.2f} "
        f"neutral F1="
        f"{fold['baseline_neutral_f1']:.4f} -> "
        f"{fold['candidate_neutral_f1']:.4f}"
    )

print("\nFULL-VALIDATION FIT (OPTIMISTIC)")
print(f"Neutral threshold: {full_threshold:.2f}")
print(f"Macro F1: {in_sample['macro_f1']:.4f}")
print(f"Neutral F1: {in_sample['neutral_f1']:.4f}")

print(f"\nSaved: {OUTPUT}")
print("Model, API and held-out test remain unchanged.")
