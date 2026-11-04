import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "roberta_goemotions_multilabel_real_v001"

DATA_DIR = ROOT / "data/processed/goemotions_v001"
METRICS_PATH = ROOT / "ml/artifacts/metrics" / f"{EXPERIMENT}_validation.json"
PREDICTIONS_PATH = ROOT / "ml/artifacts/predictions" / f"{EXPERIMENT}_validation.jsonl"
OUTPUT_PATH = ROOT / "reports/goemotions" / f"{EXPERIMENT}_rare_label_diagnostics.json"

TARGET_LABELS = (
    "grief",
    "nervousness",
    "pride",
    "relief",
)

# Diagnostic only; does not replace the model threshold.
THRESHOLDS = np.arange(1, 51, dtype=np.float64) / 100


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def score_summary(values):
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return None

    return {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "p90": float(np.percentile(values, 90)),
        "max": float(values.max()),
    }


def threshold_analysis(truth, scores, thresholds):
    results = []

    for threshold in thresholds:
        predicted = scores >= threshold

        precision, recall, f1, _ = precision_recall_fscore_support(
            truth,
            predicted,
            average="binary",
            zero_division=0,
        )

        results.append(
            {
                "threshold": float(round(threshold, 4)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "predicted_positive": int(predicted.sum()),
                "true_positive": int(np.count_nonzero(predicted & truth)),
                "false_positive": int(np.count_nonzero(predicted & ~truth)),
            }
        )

    # Deterministic tie-break: higher threshold.
    best = max(
        results,
        key=lambda row: (row["f1"], row["threshold"]),
    )

    return best, results


def analyze_label(index, names, truth, probabilities, global_threshold):
    positive = truth[:, index].astype(bool)
    negative = ~positive
    scores = probabilities[:, index]

    if not positive.any() or not negative.any():
        raise ValueError(f"Cannot analyze label without both classes: {names[index]}")

    best, sweep = threshold_analysis(
        positive,
        scores,
        THRESHOLDS,
    )

    global_predictions = scores >= global_threshold

    alternative_counts = []

    for other_index, name in enumerate(names):
        if other_index == index:
            continue

        count = int(
            np.count_nonzero(
                (probabilities[:, other_index] >= global_threshold) & positive
            )
        )

        alternative_counts.append(
            {
                "label": name,
                "count": count,
            }
        )

    alternative_counts.sort(key=lambda item: (-item["count"], item["label"]))

    ap = float(average_precision_score(positive, scores))
    prevalence = float(positive.mean())

    return {
        "label": names[index],
        "support": int(positive.sum()),
        "prevalence": prevalence,
        "average_precision": ap,
        "ap_over_prevalence": (float(ap / prevalence) if prevalence > 0 else None),
        "positive_scores": score_summary(scores[positive]),
        "negative_scores": score_summary(scores[negative]),
        "positive_scores_at_or_above_global_threshold": int(
            np.count_nonzero(scores[positive] >= global_threshold)
        ),
        "global_predicted_positive": int(global_predictions.sum()),
        "best_diagnostic_threshold": best,
        "threshold_sweep": sweep,
        "top_alternative_predictions": (alternative_counts[:8]),
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Report already exists: {OUTPUT_PATH}")

    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))

    if (
        metrics["experiment_id"] != EXPERIMENT
        or metrics["evaluation_split"] != "validation"
        or metrics["test_evaluated"] is not False
    ):
        raise ValueError("Unexpected experiment metadata")

    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if file_sha256(manifest_path) != metrics["dataset_manifest_sha256"]:
        raise ValueError("Dataset manifest changed")

    names = manifest["label_names"]
    num_labels = len(names)

    if num_labels != 28:
        raise ValueError("Expected 28 labels")

    validation_info = manifest["processed_splits"]["validation"]
    validation_path = DATA_DIR / "validation.jsonl"

    if file_sha256(validation_path) != validation_info["sha256"]:
        raise ValueError("Validation checksum mismatch")

    source_rows = list(read_jsonl(validation_path))
    prediction_rows = list(read_jsonl(PREDICTIONS_PATH))

    if len(source_rows) != validation_info["rows"] or len(prediction_rows) != len(
        source_rows
    ):
        raise ValueError("Validation row count mismatch")

    y_true = np.zeros(
        (len(source_rows), num_labels),
        dtype=np.int8,
    )

    probabilities = np.asarray(
        [row["probabilities"] for row in prediction_rows],
        dtype=np.float64,
    )

    if (
        probabilities.shape != y_true.shape
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Invalid probability matrix")

    global_threshold = float(metrics["threshold"])

    if not 0 < global_threshold < 1:
        raise ValueError("Invalid global threshold")

    seen_ids = set()

    for i, (source, prediction) in enumerate(zip(source_rows, prediction_rows)):
        example_id = source["example_id"]

        if example_id in seen_ids:
            raise ValueError("Duplicate validation ID")

        seen_ids.add(example_id)

        if (
            prediction["example_id"] != example_id
            or prediction["source_id"] != source["source_id"]
            or sorted(prediction["true_labels"]) != sorted(source["labels"])
        ):
            raise ValueError(f"Prediction alignment mismatch at row {i}")

        y_true[i, source["labels"]] = 1

        expected_labels = np.flatnonzero(probabilities[i] >= global_threshold).tolist()

        if sorted(prediction["predicted_labels"]) != expected_labels:
            raise ValueError(f"Prediction threshold mismatch at row {i}")

    y_pred = (probabilities >= global_threshold).astype(np.int8)

    macro_f1 = float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )

    if not np.isclose(macro_f1, metrics["macro_f1"], atol=1e-10):
        raise ValueError("Saved macro F1 mismatch")

    analysis = {
        "analysis_id": f"{EXPERIMENT}_rare_label_diagnostics",
        "experiment_id": EXPERIMENT,
        "split": "validation",
        "examples": len(source_rows),
        "global_threshold": global_threshold,
        "diagnostic_threshold_grid": [float(round(x, 4)) for x in THRESHOLDS],
        "labels": {},
        "limitations": [
            "Threshold sweep is diagnostic, not promoted.",
            "Best per-label F1 is selected and measured on the same validation data.",
            "Rare-label AP is uncertain with small support.",
            "No test data were accessed.",
        ],
    }

    for index, label in enumerate(names):
        analysis["labels"][label] = analyze_label(
            index,
            names,
            y_true,
            probabilities,
            global_threshold,
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Experiment: {EXPERIMENT}")
    print(f"Validation examples: {len(source_rows)}")
    print(f"Global threshold: {global_threshold:.2f}")
    print(f"Macro F1 reproduced: {macro_f1:.4f}")

    print("\nRare-label diagnostics:")

    for label in TARGET_LABELS:
        item = analysis["labels"][label]
        best = item["best_diagnostic_threshold"]

        print(f"\n{label.upper()}")
        print(f"  Support: {item['support']}")
        print(f"  Prevalence: {item['prevalence']:.4f}")
        print(f"  Average precision: {item['average_precision']:.4f}")
        print(f"  AP / prevalence: {item['ap_over_prevalence']:.2f}")
        print(f"  Positive score median: {item['positive_scores']['median']:.4f}")
        print(f"  Positive score max: {item['positive_scores']['max']:.4f}")
        print(f"  Negative score median: {item['negative_scores']['median']:.4f}")
        print(f"  Best diagnostic threshold: {best['threshold']:.2f}")
        print(
            f"  Diagnostic F1: {best['f1']:.4f}, "
            f"precision: {best['precision']:.4f}, "
            f"recall: {best['recall']:.4f}"
        )

        alternatives = [
            f"{x['label']}={x['count']}"
            for x in item["top_alternative_predictions"][:5]
        ]
        print("  Alternative predictions: " + ", ".join(alternatives))

    print(f"\n[OK] Diagnostics saved: {OUTPUT_PATH}")
    print("[OK] P11-D5 rare-label analysis completed")


if __name__ == "__main__":
    main()
