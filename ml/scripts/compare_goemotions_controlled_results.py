import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)

from ml.src.data.goemotions_dataset import load_goemotions_rows


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = {
    "uniform": "roberta_goemotions_uniform_v001",
    "rare4x": "roberta_goemotions_rare4x_v001",
    "synthetic": "roberta_goemotions_synthetic_v002",
    "hybrid": "roberta_goemotions_hybrid_v001",
}

TARGET_LABELS = (
    "grief",
    "nervousness",
    "pride",
    "relief",
)

OUTPUT_PATH = ROOT / "reports/goemotions/controlled_models_comparison_v003.json"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_ground_truth(rows, num_labels):
    truth = np.zeros(
        (len(rows), num_labels),
        dtype=np.int8,
    )

    for index, row in enumerate(rows):
        truth[index, row["labels"]] = 1

    return truth


def load_experiment(name, experiment, rows, truth):
    metrics_path = ROOT / f"ml/artifacts/metrics/{experiment}_validation.json"

    predictions_path = ROOT / f"ml/artifacts/predictions/{experiment}_validation.jsonl"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = read_jsonl(predictions_path)

    if (
        metrics["experiment_id"] != experiment
        or metrics["evaluation_split"] != "validation"
        or metrics["test_evaluated"] is not False
    ):
        raise ValueError(f"{name}: unexpected experiment metadata")

    if len(predictions) != len(rows):
        raise ValueError(f"{name}: validation row count mismatch")

    probabilities = np.asarray(
        [row["probabilities"] for row in predictions],
        dtype=np.float64,
    )

    if probabilities.shape != truth.shape:
        raise ValueError(f"{name}: probability shape mismatch")

    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError(f"{name}: invalid probabilities")

    threshold = float(metrics["selected_threshold"])

    if not 0 < threshold < 1:
        raise ValueError(f"{name}: invalid threshold")

    predicted = (probabilities >= threshold).astype(np.int8)

    for index, (source, prediction) in enumerate(zip(rows, predictions)):
        if (
            source["example_id"] != prediction["example_id"]
            or source["source_id"] != prediction["source_id"]
            or sorted(source["labels"]) != sorted(prediction["true_labels"])
            or np.flatnonzero(predicted[index]).tolist()
            != sorted(prediction["predicted_labels"])
        ):
            raise ValueError(f"{name}: alignment mismatch at {index}")

    macro_f1 = float(
        f1_score(
            truth,
            predicted,
            average="macro",
            zero_division=0,
        )
    )

    micro_f1 = float(
        f1_score(
            truth,
            predicted,
            average="micro",
            zero_division=0,
        )
    )

    macro_ap = float(
        average_precision_score(
            truth,
            probabilities,
            average="macro",
        )
    )

    for metric_name, calculated in (
        ("macro_f1", macro_f1),
        ("micro_f1", micro_f1),
        ("macro_average_precision", macro_ap),
    ):
        saved = float(metrics[metric_name])

        if not np.isclose(
            calculated,
            saved,
            atol=1e-9,
            rtol=1e-9,
        ):
            raise ValueError(
                f"{name}: {metric_name} mismatch ({calculated} != {saved})"
            )

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        average=None,
        zero_division=0,
    )

    return {
        "experiment_id": experiment,
        "threshold": threshold,
        "probabilities": probabilities,
        "predicted": predicted,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "macro_ap": macro_ap,
        "prediction_sha256": sha256_file(predictions_path),
        "metrics_sha256": sha256_file(metrics_path),
        "optimizer_steps": metrics["optimizer_steps"],
    }


def summarize_model(result, truth, label_names):
    predicted = result["predicted"]
    probabilities = result["probabilities"]

    true_cardinality = truth.sum(axis=1)
    predicted_cardinality = predicted.sum(axis=1)

    per_label = {}

    for index, label in enumerate(label_names):
        true_positive = int(
            np.count_nonzero((truth[:, index] == 1) & (predicted[:, index] == 1))
        )

        false_positive = int(
            np.count_nonzero((truth[:, index] == 0) & (predicted[:, index] == 1))
        )

        false_negative = int(
            np.count_nonzero((truth[:, index] == 1) & (predicted[:, index] == 0))
        )

        per_label[label] = {
            "support": int(result["support"][index]),
            "predicted_positive": int(predicted[:, index].sum()),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": float(result["precision"][index]),
            "recall": float(result["recall"][index]),
            "f1": float(result["f1"][index]),
            "average_precision": float(
                average_precision_score(
                    truth[:, index],
                    probabilities[:, index],
                )
            ),
        }

    return {
        "experiment_id": result["experiment_id"],
        "threshold": result["threshold"],
        "macro_f1": result["macro_f1"],
        "micro_f1": result["micro_f1"],
        "macro_average_precision": result["macro_ap"],
        "exact_match_accuracy": float(accuracy_score(truth, predicted)),
        "example_based_f1": float(
            f1_score(
                truth,
                predicted,
                average="samples",
                zero_division=0,
            )
        ),
        "mean_true_labels": float(true_cardinality.mean()),
        "mean_predicted_labels": float(predicted_cardinality.mean()),
        "zero_label_predictions": int(np.count_nonzero(predicted_cardinality == 0)),
        "multi_label_predictions": int(np.count_nonzero(predicted_cardinality > 1)),
        "predicted_cardinality_distribution": {
            str(key): int(value)
            for key, value in sorted(Counter(predicted_cardinality.tolist()).items())
        },
        "optimizer_steps": result["optimizer_steps"],
        "prediction_sha256": (result["prediction_sha256"]),
        "metrics_sha256": result["metrics_sha256"],
        "per_label": per_label,
    }


def compare_models(models, label_names):
    comparisons = {}

    for name in ("rare4x", "synthetic", "hybrid"):
        current = models[name]
        control = models["uniform"]

        changes = []

        for label in label_names:
            a = current["per_label"][label]
            b = control["per_label"][label]

            changes.append(
                {
                    "label": label,
                    "support": a["support"],
                    "f1_delta": a["f1"] - b["f1"],
                    "ap_delta": (a["average_precision"] - b["average_precision"]),
                    "precision_delta": (a["precision"] - b["precision"]),
                    "recall_delta": (a["recall"] - b["recall"]),
                }
            )

        comparisons[f"{name}_vs_uniform"] = {
            "macro_f1_delta": (current["macro_f1"] - control["macro_f1"]),
            "micro_f1_delta": (current["micro_f1"] - control["micro_f1"]),
            "macro_ap_delta": (
                current["macro_average_precision"] - control["macro_average_precision"]
            ),
            "per_label_changes": sorted(
                changes,
                key=lambda row: row["f1_delta"],
                reverse=True,
            ),
        }

    return comparisons


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Comparison already exists: {OUTPUT_PATH}")

    rows, label_names = load_goemotions_rows("validation")

    if len(rows) != 5426 or len(label_names) != 28:
        raise ValueError("Unexpected validation dataset")

    truth = build_ground_truth(rows, len(label_names))

    raw_results = {
        name: load_experiment(
            name,
            experiment,
            rows,
            truth,
        )
        for name, experiment in EXPERIMENTS.items()
    }

    step_counts = {result["optimizer_steps"] for result in raw_results.values()}

    if step_counts != {5408}:
        raise ValueError(f"Optimizer budgets differ: {step_counts}")

    models = {
        name: summarize_model(
            result,
            truth,
            label_names,
        )
        for name, result in raw_results.items()
    }

    report = {
        "analysis_id": "controlled_models_comparison_v001",
        "split": "validation",
        "validation_examples": len(rows),
        "test_evaluated": False,
        "target_labels": list(TARGET_LABELS),
        "models": models,
        "comparisons": compare_models(models, label_names),
        "limitations": [
            "Results are from one seed per condition.",
            "Thresholds were optimized on validation.",
            "Synthetic labels were LLM-proposed.",
            "This analysis does not establish statistical significance.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(
            report,
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")

    print("\nP11-E4 — CONTROLLED MODEL COMPARISON")
    print("-" * 65)

    for name, item in models.items():
        print(f"\n{name.upper()}")
        print(f"  Macro F1:   {item['macro_f1']:.4f}")
        print(f"  Micro F1:   {item['micro_f1']:.4f}")
        print(f"  Macro AP:   {item['macro_average_precision']:.4f}")
        print(f"  Exact match: {item['exact_match_accuracy']:.4f}")
        print(f"  Example F1: {item['example_based_f1']:.4f}")
        print(f"  Mean labels: {item['mean_predicted_labels']:.3f}")

    print("\nRARE-LABEL RESULTS")

    print("-" * 78)

    names = ("uniform", "rare4x", "synthetic", "hybrid")

    print(f"{'Label':<14}" + "".join(f"{name:>14}" for name in names))

    for label in TARGET_LABELS:
        print(
            f"{label:<14}"
            + "".join(
                f"{models[name]['per_label'][label]['f1']:>14.4f}" for name in names
            )
        )

    print("\nLARGEST F1 CHANGES VS UNIFORM")

    for comparison, details in report["comparisons"].items():
        changes = details["per_label_changes"]

        print(f"\n{comparison}")

        print("  Top improvements:")
        for row in changes[:5]:
            print(f"    {row['label']:<16} {row['f1_delta']:+.4f}")

        print("  Top regressions:")
        for row in changes[-5:]:
            print(f"    {row['label']:<16} {row['f1_delta']:+.4f}")

    print(f"\n[OK] Report saved: {OUTPUT_PATH}")
    print("[OK] P11-E4 comparison complete")


if __name__ == "__main__":
    main()
