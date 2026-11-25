import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / (
    "ml/artifacts/predictions/roberta_goemotions_rare4x_v001_validation.jsonl"
)
METRICS = ROOT / ("ml/artifacts/metrics/roberta_goemotions_rare4x_v001_validation.json")
OUTPUT = ROOT / ("reports/goemotions/optimization/opt003_neutral_diagnostics.json")

THRESHOLD = 0.25

reference = json.loads(METRICS.read_text(encoding="utf-8"))
labels = list(reference["per_label"])
neutral_id = labels.index("neutral")

with INPUT.open(encoding="utf-8") as file:
    rows = [json.loads(line) for line in file if line.strip()]

n = len(rows)
k = len(labels)
assert (n, k) == (5426, 28)
assert reference["threshold"] == THRESHOLD

scores = np.array([row["probabilities"] for row in rows], dtype=np.float64)
truth = np.zeros((n, k), dtype=bool)

assert scores.shape == (n, k)
assert np.isfinite(scores).all()
assert np.all((scores >= 0) & (scores <= 1))

for i, row in enumerate(rows):
    truth[i, row["true_labels"]] = True
    actual = np.flatnonzero(scores[i] >= THRESHOLD).tolist()
    assert sorted(row["predicted_labels"]) == actual

pred = scores >= THRESHOLD

true_neutral = truth[:, neutral_id]
pred_neutral = pred[:, neutral_id]

true_other = truth.copy()
true_other[:, neutral_id] = False

pred_other = pred.copy()
pred_other[:, neutral_id] = False

has_true_emotion = true_other.any(axis=1)
has_pred_emotion = pred_other.any(axis=1)

groups = {
    "neutral_only_reference": true_neutral & ~has_true_emotion,
    "neutral_and_emotion_reference": true_neutral & has_true_emotion,
    "emotion_without_neutral_reference": ~true_neutral & has_true_emotion,
    "no_reference_labels": ~truth.any(axis=1),
}


def count(mask):
    return int(np.count_nonzero(mask))


def fraction(numerator, denominator):
    return round(numerator / denominator, 6) if denominator else None


summary = {}

for name, mask in groups.items():
    total = count(mask)
    neutral_predictions = count(mask & pred_neutral)
    emotion_predictions = count(mask & has_pred_emotion)

    summary[name] = {
        "examples": total,
        "predicted_neutral": neutral_predictions,
        "predicted_neutral_rate": fraction(neutral_predictions, total),
        "predicted_any_emotion": emotion_predictions,
        "predicted_any_emotion_rate": fraction(emotion_predictions, total),
        "no_predictions": count(mask & ~pred.any(axis=1)),
        "both_neutral_and_emotion_predicted": count(
            mask & pred_neutral & has_pred_emotion
        ),
        "median_neutral_score": (
            float(np.median(scores[mask, neutral_id])) if total else None
        ),
    }

# For each non-neutral emotion, calculate conditional
# neutral false positives and false negatives.
# The denominator is restricted to examples annotated
# with that emotion, avoiding raw-frequency comparison.
per_emotion = []

for j, label in enumerate(labels):
    if j == neutral_id:
        continue

    annotated = truth[:, j]
    support = count(annotated)

    neutral_wrong = annotated & ~true_neutral & pred_neutral
    missed_emotion = annotated & ~pred[:, j]

    # These are cases where the correct emotion was missed
    # and neutral was incorrectly detected.
    substitution = missed_emotion & neutral_wrong

    # Check also whether the model confuses this label
    # with non-neutral predictions. Multiple labels possible.
    incorrect_other = pred_other & ~true_other
    incorrect_other[:, j] = False

    errors_with_other = missed_emotion & incorrect_other.any(axis=1)

    per_emotion.append(
        {
            "label": label,
            "support": support,
            "missed_count": count(missed_emotion),
            "missed_rate": fraction(count(missed_emotion), support),
            "neutral_false_positive_count": count(neutral_wrong),
            "neutral_false_positive_rate": fraction(
                count(neutral_wrong), count(annotated & ~true_neutral)
            ),
            "missed_and_neutral_fp": count(substitution),
            "missed_and_neutral_fp_rate": fraction(count(substitution), support),
            "missed_and_other_emotion_fp": count(errors_with_other),
        }
    )

# Which emotions are incorrectly detected when the
# reference is neutral-only? Count each label independently.
neutral_only = groups["neutral_only_reference"]
neutral_only_count = count(neutral_only)

false_emotions_on_neutral = []

for j, label in enumerate(labels):
    if j == neutral_id:
        continue

    occurrences = count(neutral_only & pred[:, j])
    false_emotions_on_neutral.append(
        {
            "label": label,
            "count": occurrences,
            "rate": fraction(occurrences, neutral_only_count),
        }
    )

false_emotions_on_neutral.sort(key=lambda x: x["count"], reverse=True)

neutral_fp = ~true_neutral & pred_neutral
neutral_fn = true_neutral & ~pred_neutral

report = {
    "experiment": "OPT-003",
    "model": "roberta_goemotions_rare4x_v001",
    "split": "validation",
    "threshold": THRESHOLD,
    "total_examples": n,
    "reference_groups": summary,
    "neutral_errors": {
        "false_positives": count(neutral_fp),
        "false_negatives": count(neutral_fn),
        "fp_with_annotated_other_emotion": count(neutral_fp & has_true_emotion),
        "fn_with_predicted_other_emotion": count(neutral_fn & has_pred_emotion),
    },
    "per_emotion": per_emotion,
    "false_emotions_on_neutral_only": false_emotions_on_neutral,
    "notes": [
        "Validation-only exploratory analysis.",
        "Reference neutral is not forced to be exclusive.",
        "Some event counts overlap across emotion labels.",
        "Rates are conditional on explicitly stated groups.",
        "No threshold or production behavior changed.",
        "No held-out predictions accessed.",
    ],
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nOPT-003 — NEUTRAL-LABEL DIAGNOSTICS")
print(f"Examples: {n} | Neutral label ID: {neutral_id}")

print("\nREFERENCE GROUPS")
for name, group in summary.items():
    print(
        f"{name:39} n={group['examples']:4} "
        f"neutral_pred={group['predicted_neutral_rate']} "
        f"emotion_pred={group['predicted_any_emotion_rate']}"
    )

print("\nNEUTRAL ERRORS")
for key, value in report["neutral_errors"].items():
    print(f"{key:39} {value}")

print("\nHIGHEST MISSED-EMOTION + FALSE NEUTRAL")
for item in sorted(
    per_emotion,
    key=lambda x: x["missed_and_neutral_fp"],
    reverse=True,
)[:12]:
    print(
        f"{item['label']:18} "
        f"support={item['support']:4} "
        f"missed+neutral={item['missed_and_neutral_fp']:4} "
        f"rate={item['missed_and_neutral_fp_rate']}"
    )

print("\nFALSE EMOTIONS ON NEUTRAL-ONLY REFERENCES")
for item in false_emotions_on_neutral[:12]:
    print(f"{item['label']:18} count={item['count']:4} rate={item['rate']}")

print(f"\nSaved: {OUTPUT}")
print("No model or serving changes performed.")
