import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS = (
    ROOT
    / "ml/artifacts/predictions"
    / "roberta_goemotions_rare4x_v001_validation.jsonl"
)

METRICS = (
    ROOT / "ml/artifacts/metrics" / "roberta_goemotions_rare4x_v001_validation.json"
)

OUTPUT = ROOT / "reports/goemotions/optimization" / "opt002_error_diagnostics.json"

THRESHOLD = 0.25

# Load existing validation evidence.
reference = json.loads(METRICS.read_text(encoding="utf-8"))
labels = list(reference["per_label"].keys())

with PREDICTIONS.open(encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

assert len(labels) == 28
assert len(rows) == 5426
assert reference["threshold"] == THRESHOLD

scores = np.asarray(
    [row["probabilities"] for row in rows],
    dtype=np.float64,
)

truth = np.zeros(scores.shape, dtype=bool)

assert scores.shape == (5426, 28)
assert np.isfinite(scores).all()
assert ((scores >= 0) & (scores <= 1)).all()
assert len({row["example_id"] for row in rows}) == len(rows)

for index, row in enumerate(rows):
    ids = row["true_labels"]
    assert len(ids) == len(set(ids))
    assert all(isinstance(x, int) and 0 <= x < 28 for x in ids)

    truth[index, ids] = True

predicted = scores >= THRESHOLD

for index, row in enumerate(rows):
    actual = np.flatnonzero(predicted[index]).tolist()
    assert sorted(row["predicted_labels"]) == actual, (
        f"Saved prediction mismatch: {row['example_id']}"
    )

tp = truth & predicted
fp = ~truth & predicted
fn = truth & ~predicted


def safe_f1(true_positives, false_positives, false_negatives):
    denominator = 2 * true_positives + false_positives + false_negatives
    return 2 * true_positives / denominator if denominator else 0.0


def median(values):
    return float(np.median(values)) if len(values) else None


details = []

for j, name in enumerate(labels):
    support = int(truth[:, j].sum())
    true_pos = int(tp[:, j].sum())
    false_pos = int(fp[:, j].sum())
    false_neg = int(fn[:, j].sum())

    positives = scores[truth[:, j], j]
    negatives = scores[~truth[:, j], j]

    details.append(
        {
            "label": name,
            "support": support,
            "tp": true_pos,
            "fp": false_pos,
            "fn": false_neg,
            "precision": (
                true_pos / (true_pos + false_pos) if true_pos + false_pos else 0.0
            ),
            "recall": true_pos / support if support else 0.0,
            "f1": safe_f1(true_pos, false_pos, false_neg),
            "average_precision": float(
                reference["per_label"][name]["average_precision"]
            ),
            "positive_score_median": median(positives),
            "negative_score_median": median(negatives),
            "missed_positive_score_median": median(scores[fn[:, j], j]),
            "false_positive_score_median": median(scores[fp[:, j], j]),
        }
    )

    assert support == reference["per_label"][name]["support"]
    assert abs(details[-1]["f1"] - reference["per_label"][name]["f1"]) < 1e-10

# Only count pairs where both sides are genuine errors.
# A correctly predicted co-occurring label is NOT a confusion.
pair_counts = Counter()

for i in range(len(rows)):
    missed = np.flatnonzero(fn[i])
    incorrect = np.flatnonzero(fp[i])

    for true_id in missed:
        for wrong_id in incorrect:
            pair_counts[(int(true_id), int(wrong_id))] += 1

pairs = [
    {
        "missed_true_label": labels[a],
        "incorrect_predicted_label": labels[b],
        "count": count,
        "share_of_missed_label": (
            count / details[a]["fn"] if details[a]["fn"] else 0.0
        ),
    }
    for (a, b), count in pair_counts.most_common()
]

# Aggregate prediction-set diagnostics.
exact_match = float(np.mean(np.all(truth == predicted, axis=1)))
no_detections = int(np.sum(~predicted.any(axis=1)))
multiple_detections = int(np.sum(predicted.sum(axis=1) > 1))

diagnostics = {
    "experiment": "OPT-002",
    "split": "validation",
    "model": "roberta_goemotions_rare4x_v001",
    "threshold": THRESHOLD,
    "examples": len(rows),
    "labels": len(labels),
    "aggregate": {
        "total_true_positives": int(tp.sum()),
        "total_false_positives": int(fp.sum()),
        "total_false_negatives": int(fn.sum()),
        "exact_match_accuracy": exact_match,
        "no_detection_examples": no_detections,
        "multiple_detection_examples": multiple_detections,
    },
    "per_label": details,
    "cross_label_errors": pairs,
    "notes": [
        "Validation-only exploratory diagnostics.",
        "Cross-label pairs count missed true labels alongside "
        "incorrectly detected labels, not exclusive-class confusion.",
        "Each example can contribute multiple cross-label pairs.",
        "Positive/negative score medians are descriptive, not calibrated.",
        "No held-out test predictions were accessed.",
    ],
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(diagnostics, indent=2),
    encoding="utf-8",
)

print("\nOPT-002 — MULTI-LABEL ERROR DIAGNOSTICS")
print(f"Validation examples: {len(rows)}")
print(f"Exact match: {exact_match:.4f}")
print(f"No detections: {no_detections}")
print(f"Multiple detections: {multiple_detections}")
print(f"TP={tp.sum()}  FP={fp.sum()}  FN={fn.sum()}")

print("\nPER-LABEL ERRORS")
print(
    f"{'Label':18} {'Sup':>5} {'FP':>5} {'FN':>5} "
    f"{'F1':>7} {'AP':>7} {'PosMed':>8} {'NegMed':>8}"
)

for item in sorted(details, key=lambda x: x["f1"]):
    print(
        f"{item['label']:18} "
        f"{item['support']:5} "
        f"{item['fp']:5} "
        f"{item['fn']:5} "
        f"{item['f1']:7.3f} "
        f"{item['average_precision']:7.3f} "
        f"{item['positive_score_median']:8.3f} "
        f"{item['negative_score_median']:8.3f}"
    )

print("\nTOP 15 CROSS-LABEL ERROR PAIRS")
print(f"{'Missed true label':19} {'Wrongly detected':19} {'Count':>6}")

for item in pairs[:15]:
    print(
        f"{item['missed_true_label']:19} "
        f"{item['incorrect_predicted_label']:19} "
        f"{item['count']:6}"
    )

print(f"\nSaved aggregate diagnostics: {OUTPUT}")
print("No model weights, serving settings or test data modified.")
