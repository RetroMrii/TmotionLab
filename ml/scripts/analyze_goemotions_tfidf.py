import json
from collections import Counter
from pathlib import Path
import argparse
import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = (
    "tfidf_goemotions_multilabel_v001",
    "roberta_goemotions_multilabel_real_v001",
)


def artifact_paths(experiment):
    return (
        ROOT / "ml/artifacts/metrics" / f"{experiment}_validation.json",
        ROOT / "ml/artifacts/predictions" / f"{experiment}_validation.jsonl",
        ROOT / "reports/goemotions" / f"{experiment}_diagnostics.json",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        required=True,
        choices=EXPERIMENTS,
    )
    args = parser.parse_args()
    experiment = args.experiment

    metrics_path, predictions_path, report_path = artifact_paths(experiment)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    with predictions_path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    if len(rows) != 5426:
        raise ValueError("Unexpected validation row count")

    ids = [row["example_id"] for row in rows]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate example IDs")

    label_names = list(metrics["per_label"])
    num_labels = len(label_names)

    if num_labels != 28:
        raise ValueError("Expected 28 labels")

    y_true = np.zeros((len(rows), num_labels), dtype=np.int8)

    y_pred = np.zeros_like(y_true)

    probabilities = np.asarray(
        [row["probabilities"] for row in rows],
        dtype=np.float64,
    )

    if probabilities.shape != y_true.shape:
        raise ValueError("Invalid probability shape")

    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Invalid probabilities")

    threshold = metrics["threshold"]

    for i, row in enumerate(rows):
        y_true[i, row["true_labels"]] = 1
        y_pred[i, row["predicted_labels"]] = 1

    expected_predictions = (probabilities >= threshold).astype(np.int8)

    if not np.array_equal(y_pred, expected_predictions):
        raise ValueError("Saved predictions disagree with selected threshold")

    actual_macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if not np.isclose(actual_macro_f1, metrics["macro_f1"], atol=1e-10):
        raise ValueError("Saved macro F1 mismatch")

    true_cardinality = Counter(y_true.sum(axis=1).tolist())
    predicted_cardinality = Counter(y_pred.sum(axis=1).tolist())

    per_label = []

    for i, name in enumerate(label_names):
        truth = y_true[:, i]
        prediction = y_pred[:, i]

        tp = int(np.sum((truth == 1) & (prediction == 1)))
        fp = int(np.sum((truth == 0) & (prediction == 1)))
        fn = int(np.sum((truth == 1) & (prediction == 0)))

        per_label.append(
            {
                "label": name,
                "support": int(truth.sum()),
                "predicted_positive": int(prediction.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": metrics["per_label"][name]["precision"],
                "recall": metrics["per_label"][name]["recall"],
                "f1": metrics["per_label"][name]["f1"],
                "average_precision": metrics["per_label"][name]["average_precision"],
            }
        )

    per_label.sort(key=lambda item: item["f1"])

    result = {
        "experiment_id": experiment,
        "split": "validation",
        "threshold": threshold,
        "macro_f1": actual_macro_f1,
        "true_cardinality": dict(sorted(true_cardinality.items())),
        "predicted_cardinality": dict(sorted(predicted_cardinality.items())),
        "mean_true_labels": float(y_true.sum(axis=1).mean()),
        "mean_predicted_labels": float(y_pred.sum(axis=1).mean()),
        "zero_prediction_count": int(np.sum(y_pred.sum(axis=1) == 0)),
        "multi_prediction_count": int(np.sum(y_pred.sum(axis=1) > 1)),
        "per_label": per_label,
    }

    if report_path.exists():
        raise FileExistsError(f"Report already exists: {report_path}")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Threshold: {threshold:.2f}")
    print(f"Macro F1: {actual_macro_f1:.4f}")
    print(f"Mean true labels: {result['mean_true_labels']:.3f}")
    print(f"Mean predicted labels: {result['mean_predicted_labels']:.3f}")
    print(f"Zero-label predictions: {result['zero_prediction_count']}")
    print(f"Multi-label predictions: {result['multi_prediction_count']}")

    print("\nPredicted label cardinality:")
    for count, frequency in sorted(predicted_cardinality.items()):
        print(f"  {count} labels: {frequency}")

    print("\nWeakest 10 labels by F1:")
    for item in per_label[:10]:
        print(
            f"  {item['label']:<16} "
            f"support={item['support']:<5} "
            f"predicted={item['predicted_positive']:<5} "
            f"F1={item['f1']:.4f}"
        )

    print("\nStrongest 5 labels by F1:")
    for item in per_label[-5:]:
        print(f"  {item['label']:<16} F1={item['f1']:.4f}")

    print(f"\n[OK] Diagnostics saved: {report_path}")


if __name__ == "__main__":
    main()
