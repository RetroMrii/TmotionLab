import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / (
    "ml/artifacts/predictions/roberta_goemotions_rare4x_v001_validation.jsonl"
)
METRICS = ROOT / ("ml/artifacts/metrics/roberta_goemotions_rare4x_v001_validation.json")
OUTPUT = ROOT / ("reports/goemotions/optimization/opt005_conditional_neutral.json")

BASE = 0.25
GRID = np.round(np.arange(0.10, 0.601, 0.01), 2)

with PRED.open(encoding="utf-8") as handle:
    rows = [json.loads(x) for x in handle if x.strip()]

reference = json.loads(METRICS.read_text(encoding="utf-8"))
labels = list(reference["per_label"])
neutral = labels.index("neutral")

assert len(rows) == 5426
assert len(labels) == 28
assert reference["threshold"] == BASE

scores = np.asarray([r["probabilities"] for r in rows], dtype=float)
truth = np.zeros(scores.shape, dtype=bool)

assert scores.shape == (5426, 28)
assert np.isfinite(scores).all()
assert ((scores >= 0) & (scores <= 1)).all()
assert len({r["example_id"] for r in rows}) == len(rows)

for i, row in enumerate(rows):
    truth[i, row["true_labels"]] = True
    assert sorted(row["predicted_labels"]) == (
        np.flatnonzero(scores[i] >= BASE).tolist()
    )

other = np.arange(28) != neutral

# Group membership uses model predictions only.
# No ground-truth labels are used at inference time.
has_other_prediction = (scores[:, other] >= BASE).any(axis=1)


def predict(index, threshold_with, threshold_without):
    selected = scores[index]
    result = selected >= BASE
    threshold = np.where(
        has_other_prediction[index],
        threshold_with,
        threshold_without,
    )
    result[:, neutral] = selected[:, neutral] >= threshold
    return result


def evaluate(index, predictions):
    y = truth[index]
    tn = y[:, neutral]
    pn = predictions[:, neutral]

    return {
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, predictions, average="micro", zero_division=0)),
        "neutral_f1": float(f1_score(tn, pn, zero_division=0)),
        "neutral_fp": int((~tn & pn).sum()),
        "neutral_fn": int((tn & ~pn).sum()),
    }


def choose_thresholds(index):
    y = truth[index, neutral]
    p = scores[index, neutral]
    g = has_other_prediction[index]

    # Candidate decisions for each group.
    candidate = p[:, None] >= GRID[None, :]

    def counts(mask):
        positives = y[mask, None]
        decisions = candidate[mask]

        tp = (decisions & positives).sum(axis=0)
        fp = (decisions & ~positives).sum(axis=0)
        fn = (~decisions & positives).sum(axis=0)
        return tp, fp, fn

    tp_a, fp_a, fn_a = counts(g)
    tp_b, fp_b, fn_b = counts(~g)

    # Joint optimization of neutral F1.
    # Other 27 labels remain fixed, so neutral F1
    # and macro F1 have identical maximizers.
    tp = tp_a[:, None] + tp_b[None, :]
    fp = fp_a[:, None] + fp_b[None, :]
    fn = fn_a[:, None] + fn_b[None, :]

    denom = 2 * tp + fp + fn
    f1 = np.divide(
        2 * tp,
        denom,
        out=np.zeros_like(tp, dtype=float),
        where=denom > 0,
    )

    best = np.argwhere(np.isclose(f1, np.max(f1), atol=1e-12))

    # Prefer smaller changes from the existing
    # threshold if multiple configurations tie.
    distances = abs(GRID[best[:, 0]] - BASE) + abs(GRID[best[:, 1]] - BASE)
    i, j = best[np.argmin(distances)]

    return float(GRID[i]), float(GRID[j])


indices = np.arange(len(rows))
base_pred = predict(indices, BASE, BASE)
global_021_pred = predict(indices, 0.21, 0.21)

oof_pred = base_pred.copy()
fold_results = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (fit, assess) in enumerate(
    cv.split(scores, truth[:, neutral].astype(int)),
    start=1,
):
    with_t, without_t = choose_thresholds(fit)
    candidate = predict(assess, with_t, without_t)
    oof_pred[assess] = candidate

    fold_results.append(
        {
            "fold": fold,
            "with_emotion_threshold": with_t,
            "without_emotion_threshold": without_t,
            "baseline": evaluate(assess, base_pred[assess]),
            "conditional": evaluate(assess, candidate),
        }
    )

# All non-neutral predictions must be identical.
assert np.array_equal(oof_pred[:, other], base_pred[:, other])

baseline = evaluate(indices, base_pred)
constant_021 = evaluate(indices, global_021_pred)
conditional_oof = evaluate(indices, oof_pred)

assert abs(baseline["macro_f1"] - reference["macro_f1"]) < 1e-10
assert abs(baseline["micro_f1"] - reference["micro_f1"]) < 1e-10

full_a, full_b = choose_thresholds(indices)
in_sample = evaluate(indices, predict(indices, full_a, full_b))

result = {
    "experiment": "OPT-005",
    "model": "roberta_goemotions_rare4x_v001",
    "split": "validation",
    "folds": fold_results,
    "group_counts": {
        "with_other_prediction": int(has_other_prediction.sum()),
        "without_other_prediction": int((~has_other_prediction).sum()),
    },
    "baseline_025": baseline,
    "constant_021": constant_021,
    "conditional_out_of_fold": conditional_oof,
    "full_validation_fit": {
        "threshold_with_other": full_a,
        "threshold_without_other": full_b,
        "metrics": in_sample,
        "status": "optimistic_in_sample",
    },
    "notes": [
        "Only the neutral decision rule changes.",
        "The other 27 labels retain threshold 0.25.",
        "Reference neutral/emotion overlap remains allowed.",
        "Thresholds fitted within each validation fold.",
        "Held-out test predictions were not accessed.",
        "No production modifications performed.",
    ],
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

print("\nOPT-005 — CONDITIONAL NEUTRAL THRESHOLD")
print("Group counts:", result["group_counts"])

print("\nMETRIC COMPARISON")
for name, data in [
    ("Global 0.25", baseline),
    ("Fixed 0.21", constant_021),
    ("Conditional OOF", conditional_oof),
]:
    print(
        f"{name:17} "
        f"macro={data['macro_f1']:.4f} "
        f"micro={data['micro_f1']:.4f} "
        f"neutral={data['neutral_f1']:.4f} "
        f"FP={data['neutral_fp']:4} "
        f"FN={data['neutral_fn']:4}"
    )

print("\nFOLD THRESHOLDS")
for fold in fold_results:
    print(
        f"Fold {fold['fold']}: "
        f"with_emotion="
        f"{fold['with_emotion_threshold']:.2f}, "
        f"without_emotion="
        f"{fold['without_emotion_threshold']:.2f}, "
        f"neutral F1="
        f"{fold['conditional']['neutral_f1']:.4f}"
    )

print("\nFULL-VALIDATION FIT (OPTIMISTIC)")
print(f"With another emotion: {full_a:.2f}, without: {full_b:.2f}")

print(f"\nSaved: {OUTPUT}")
print("No model, serving, or held-out test changes.")
