import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "ml/artifacts/optimization/opt007"
OUT = ROOT / ("reports/goemotions/optimization/opt008_weighted_comparison.json")
REF = ROOT / ("ml/artifacts/metrics/roberta_goemotions_rare4x_v001_validation.json")

THRESHOLD = 0.25
GRID = np.round(np.arange(0.05, 0.751, 0.01), 2)
MIN_SUPPORT = 20


def load_predictions(mode):
    path = BASE / mode / "validation_predictions.jsonl"
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


reference = json.loads(REF.read_text(encoding="utf-8"))
names = list(reference["per_label"].keys())

baseline_rows = load_predictions("baseline")
weighted_rows = load_predictions("weighted")

assert len(baseline_rows) == len(weighted_rows) == 5426
assert len(names) == 28

baseline_ids = [r["example_id"] for r in baseline_rows]
weighted_ids = [r["example_id"] for r in weighted_rows]

assert baseline_ids == weighted_ids
assert len(set(baseline_ids)) == 5426

truth = np.zeros((5426, 28), dtype=bool)

for i, (a, b) in enumerate(zip(baseline_rows, weighted_rows)):
    assert sorted(a["true_labels"]) == sorted(b["true_labels"])
    truth[i, a["true_labels"]] = True

scores = {
    "baseline": np.asarray(
        [r["probabilities"] for r in baseline_rows],
        dtype=np.float64,
    ),
    "weighted": np.asarray(
        [r["probabilities"] for r in weighted_rows],
        dtype=np.float64,
    ),
}

for values in scores.values():
    assert values.shape == truth.shape
    assert np.isfinite(values).all()
    assert np.all((values >= 0) & (values <= 1))


def evaluate(y, predictions):
    return {
        "macro_f1": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, predictions, average="micro", zero_division=0)),
    }


def fit_thresholds(y, p):
    selected = np.full(28, THRESHOLD)

    for j in range(28):
        if y[:, j].sum() < MIN_SUPPORT:
            continue

        candidate = p[:, j, None] >= GRID[None, :]
        target = y[:, j, None]

        tp = (candidate & target).sum(axis=0)
        fp = (candidate & ~target).sum(axis=0)
        fn = (~candidate & target).sum(axis=0)

        denom = 2 * tp + fp + fn

        scores_f1 = np.divide(
            2 * tp,
            denom,
            out=np.zeros(len(GRID), dtype=float),
            where=denom > 0,
        )

        best = np.flatnonzero(np.isclose(scores_f1, scores_f1.max()))
        closest = best[np.argmin(abs(GRID[best] - THRESHOLD))]
        selected[j] = GRID[closest]

    return selected


# Identical folds for both models.
folds = list(
    KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    ).split(truth)
)

results = {}
label_comparison = []

for mode, p in scores.items():
    fixed = p >= THRESHOLD
    oof = np.zeros_like(truth)
    thresholds_by_fold = []

    for fit_idx, eval_idx in folds:
        selected = fit_thresholds(truth[fit_idx], p[fit_idx])

        oof[eval_idx] = p[eval_idx] >= selected[None, :]
        thresholds_by_fold.append(selected.tolist())

    full_thresholds = fit_thresholds(truth, p)

    results[mode] = {
        "fixed_025": evaluate(truth, fixed),
        "out_of_fold": evaluate(truth, oof),
        "macro_ap": float(average_precision_score(truth, p, average="macro")),
        "fold_thresholds": thresholds_by_fold,
        "full_validation_thresholds": full_thresholds.tolist(),
        "full_validation_optimistic": evaluate(truth, p >= full_thresholds[None, :]),
    }

    for j, name in enumerate(names):
        label_comparison.append(
            {
                "mode": mode,
                "label": name,
                "support": int(truth[:, j].sum()),
                "ap": float(average_precision_score(truth[:, j], p[:, j])),
                "fixed_f1": float(
                    f1_score(
                        truth[:, j],
                        fixed[:, j],
                        zero_division=0,
                    )
                ),
                "oof_f1": float(
                    f1_score(
                        truth[:, j],
                        oof[:, j],
                        zero_division=0,
                    )
                ),
            }
        )


def label_data(mode, name):
    return next(
        row for row in label_comparison if row["mode"] == mode and row["label"] == name
    )


differences = []

for name in names:
    a = label_data("baseline", name)
    b = label_data("weighted", name)

    differences.append(
        {
            "label": name,
            "support": a["support"],
            "ap_difference": b["ap"] - a["ap"],
            "oof_f1_difference": b["oof_f1"] - a["oof_f1"],
        }
    )

differences.sort(
    key=lambda row: row["ap_difference"],
    reverse=True,
)

report = {
    "experiment": "OPT-008",
    "split": "validation",
    "examples": len(truth),
    "evaluation": "Shared 5-fold out-of-fold threshold fitting",
    "results": results,
    "per_label": label_comparison,
    "differences": differences,
    "limitations": [
        "Model checkpoints were selected using validation AP.",
        "OOF fitting isolates threshold tuning, not model selection.",
        "Full-validation thresholds are optimistic.",
        "Single training seed per model.",
        "No held-out test predictions accessed.",
        "No production model changed.",
    ],
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

print("\nOPT-008 — WEIGHTED BCE COMPARISON")
print("\nAGGREGATE RESULTS")

for mode, result in results.items():
    fixed = result["fixed_025"]
    oof = result["out_of_fold"]

    print(
        f"{mode:9} "
        f"AP={result['macro_ap']:.4f} "
        f"fixed_macro={fixed['macro_f1']:.4f} "
        f"OOF_macro={oof['macro_f1']:.4f} "
        f"fixed_micro={fixed['micro_f1']:.4f} "
        f"OOF_micro={oof['micro_f1']:.4f}"
    )

print("\nPER-LABEL AP DIFFERENCES")
print(f"{'Label':18} {'Support':>7} {'AP delta':>10} {'OOF F1 delta':>13}")

for item in differences:
    print(
        f"{item['label']:18} "
        f"{item['support']:7d} "
        f"{item['ap_difference']:+10.4f} "
        f"{item['oof_f1_difference']:+13.4f}"
    )

print(f"\nSaved: {OUT}")
print("No training or production changes.")
