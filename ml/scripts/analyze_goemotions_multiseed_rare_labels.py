import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
)

from ml.src.data.goemotions_dataset import load_goemotions_rows


ROOT = Path(__file__).resolve().parents[2]

OUTPUT = ROOT / "reports/goemotions/multiseed_rare_label_analysis_v001.json"

EXPERIMENTS = {
    "uniform": "roberta_goemotions_uniform_v001",
    "rare4x": "roberta_goemotions_rare4x_v001",
    "synthetic": "roberta_goemotions_synthetic_v002",
    "hybrid": "roberta_goemotions_hybrid_v001",
}

SEEDS = (42, 123, 456)
TARGETS = ("grief", "nervousness", "pride", "relief")


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def experiment_id(strategy, seed):
    base = EXPERIMENTS[strategy]
    return base if seed == 42 else f"{base}_seed{seed}"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def describe(values):
    return {
        "mean": float(statistics.mean(values)),
        "sample_std": float(statistics.stdev(values)),
        "by_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
    }


def load_run(strategy, seed, rows, truth):
    experiment = experiment_id(strategy, seed)

    prediction_path = ROOT / f"ml/artifacts/predictions/{experiment}_validation.jsonl"

    metrics_path = ROOT / f"ml/artifacts/metrics/{experiment}_validation.json"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    predictions = read_jsonl(prediction_path)

    if (
        metrics["experiment_id"] != experiment
        or metrics["evaluation_split"] != "validation"
        or metrics["test_evaluated"] is not False
        or metrics["optimizer_steps"] != 5408
    ):
        raise ValueError(f"Invalid run metadata: {experiment}")

    if len(predictions) != len(rows):
        raise ValueError(f"Row count mismatch: {experiment}")

    probabilities = np.asarray(
        [record["probabilities"] for record in predictions],
        dtype=np.float64,
    )

    if (
        probabilities.shape != truth.shape
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError(f"Invalid probabilities: {experiment}")

    threshold = float(metrics["selected_threshold"])
    if not 0 < threshold < 1:
        raise ValueError(f"Invalid threshold: {experiment}")

    predicted = (probabilities >= threshold).astype(np.int8)

    for index, (reference, record) in enumerate(zip(rows, predictions)):
        if (
            record["example_id"] != reference["example_id"]
            or record["source_id"] != reference["source_id"]
            or sorted(record["true_labels"]) != sorted(reference["labels"])
            or sorted(record["predicted_labels"])
            != np.flatnonzero(predicted[index]).tolist()
        ):
            raise ValueError(
                f"Prediction alignment mismatch: {experiment}, row {index}"
            )

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        average=None,
        zero_division=0,
    )

    macro_f1 = float(np.mean(f1))
    macro_ap = float(average_precision_score(truth, probabilities, average="macro"))

    if not np.isclose(macro_f1, metrics["macro_f1"], atol=1e-9) or not np.isclose(
        macro_ap,
        metrics["macro_average_precision"],
        atol=1e-9,
    ):
        raise ValueError(f"Saved metrics do not reproduce: {experiment}")

    return {
        "experiment_id": experiment,
        "threshold": threshold,
        "truth": truth,
        "predicted": predicted,
        "probabilities": probabilities,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "prediction_sha256": sha256_file(prediction_path),
        "metrics_sha256": sha256_file(metrics_path),
    }


def label_metrics(result, index):
    true = result["truth"][:, index]
    pred = result["predicted"][:, index]

    tp = int(np.count_nonzero((true == 1) & (pred == 1)))
    fp = int(np.count_nonzero((true == 0) & (pred == 1)))
    fn = int(np.count_nonzero((true == 1) & (pred == 0)))

    return {
        "support": int(result["support"][index]),
        "predicted_positives": tp + fp,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": float(result["precision"][index]),
        "recall": float(result["recall"][index]),
        "f1": float(result["f1"][index]),
        "average_precision": float(
            average_precision_score(true, result["probabilities"][:, index])
        ),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Report already exists: {OUTPUT}")

    rows, names = load_goemotions_rows("validation")

    if len(rows) != 5426 or len(names) != 28:
        raise ValueError("Unexpected validation dataset")

    truth = np.zeros((len(rows), len(names)), dtype=np.int8)

    for index, row in enumerate(rows):
        truth[index, row["labels"]] = 1

    label_ids = {name: i for i, name in enumerate(names)}

    runs = {}
    for strategy in EXPERIMENTS:
        runs[strategy] = {}

        for seed in SEEDS:
            result = load_run(strategy, seed, rows, truth)

            per_label = {
                label: label_metrics(result, label_ids[label]) for label in TARGETS
            }

            target_f1 = statistics.mean(item["f1"] for item in per_label.values())

            target_ap = statistics.mean(
                item["average_precision"] for item in per_label.values()
            )

            runs[strategy][str(seed)] = {
                "experiment_id": result["experiment_id"],
                "threshold": result["threshold"],
                "rare_macro_f1": target_f1,
                "rare_macro_ap": target_ap,
                "per_label": per_label,
                "prediction_sha256": (result["prediction_sha256"]),
                "metrics_sha256": (result["metrics_sha256"]),
            }

    summary = {}

    for strategy in EXPERIMENTS:
        summary[strategy] = {
            "rare_macro_f1": describe(
                [runs[strategy][str(seed)]["rare_macro_f1"] for seed in SEEDS]
            ),
            "rare_macro_ap": describe(
                [runs[strategy][str(seed)]["rare_macro_ap"] for seed in SEEDS]
            ),
            "per_label": {},
        }

        for label in TARGETS:
            summary[strategy]["per_label"][label] = {
                metric: describe(
                    [
                        runs[strategy][str(seed)]["per_label"][label][metric]
                        for seed in SEEDS
                    ]
                )
                for metric in (
                    "f1",
                    "precision",
                    "recall",
                    "average_precision",
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                )
            }

    paired = {}

    for strategy in ("uniform", "synthetic", "hybrid"):
        paired[strategy] = {}

        for metric in ("rare_macro_f1", "rare_macro_ap"):
            paired[strategy][metric] = describe(
                [
                    runs[strategy][str(seed)][metric]
                    - runs["rare4x"][str(seed)][metric]
                    for seed in SEEDS
                ]
            )

        paired[strategy]["per_label"] = {
            label: {
                metric: describe(
                    [
                        runs[strategy][str(seed)]["per_label"][label][metric]
                        - runs["rare4x"][str(seed)]["per_label"][label][metric]
                        for seed in SEEDS
                    ]
                )
                for metric in ("f1", "average_precision")
            }
            for label in TARGETS
        }

    report = {
        "analysis_id": "multiseed_rare_label_analysis_v001",
        "split": "validation",
        "test_evaluated": False,
        "seeds": list(SEEDS),
        "target_labels": list(TARGETS),
        "support": {label: int(truth[:, label_ids[label]].sum()) for label in TARGETS},
        "runs": runs,
        "summary": summary,
        "paired_vs_rare4x": paired,
        "limitations": [
            "Only three paired random seeds.",
            "Rare-label validation support is small.",
            "F1 thresholds were tuned on validation.",
            "AP evaluates ranking without threshold selection.",
            "Synthetic labels are LLM-proposed, not independently human-verified.",
            "The validation set has been used in exploratory treatment development.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print("P11-E6D — MULTI-SEED RARE-LABEL ANALYSIS")
    print("-" * 70)

    print("\nMEAN F1 ACROSS THE FOUR TARGET EMOTIONS")
    for strategy in EXPERIMENTS:
        item = summary[strategy]["rare_macro_f1"]
        print(f"  {strategy:<12} {item['mean']:.4f} +/- {item['sample_std']:.4f}")

    print("\nPER-LABEL MEAN F1")
    print(f"{'Emotion':<14}" + "".join(f"{strategy:>13}" for strategy in EXPERIMENTS))

    for label in TARGETS:
        print(
            f"{label:<14}"
            + "".join(
                f"{summary[strategy]['per_label'][label]['f1']['mean']:>13.4f}"
                for strategy in EXPERIMENTS
            )
        )

    print("\nHYBRID - REAL4X (PAIRED F1 DIFFERENCES)")
    for label in TARGETS:
        item = paired["hybrid"]["per_label"][label]["f1"]
        values = item["by_seed"]

        print(
            f"  {label:<13} "
            f"mean={item['mean']:+.4f} "
            f"seed42={values['42']:+.4f} "
            f"seed123={values['123']:+.4f} "
            f"seed456={values['456']:+.4f}"
        )

    for metric in ("rare_macro_f1", "rare_macro_ap"):
        item = paired["hybrid"][metric]
        print(f"\nHybrid vs real4x {metric}: {item['mean']:+.4f}")

    print(f"\n[OK] Report saved: {OUTPUT}")
    print("[OK] All 12 prediction files analyzed")


if __name__ == "__main__":
    main()
