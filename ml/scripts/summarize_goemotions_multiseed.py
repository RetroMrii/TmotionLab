import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

OUTPUT = ROOT / "reports/goemotions/multiseed_robustness_v001.json"

EXPERIMENTS = {
    "uniform": "roberta_goemotions_uniform_v001",
    "rare4x": "roberta_goemotions_rare4x_v001",
    "synthetic": "roberta_goemotions_synthetic_v002",
    "hybrid": "roberta_goemotions_hybrid_v001",
}

SEEDS = (42, 123, 456)
METRICS = (
    "macro_f1",
    "micro_f1",
    "macro_average_precision",
)
BASELINE = "uniform"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_path(experiment):
    return ROOT / f"ml/artifacts/metrics/{experiment}_validation.json"


def prediction_path(experiment):
    return ROOT / f"ml/artifacts/predictions/{experiment}_validation.jsonl"


def experiment_id(strategy, seed):
    base = EXPERIMENTS[strategy]
    return base if seed == 42 else f"{base}_seed{seed}"


def summarize(values):
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "values_by_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Report already exists: {OUTPUT}")

    runs = {}
    summary = {}
    paired = {}

    for strategy in EXPERIMENTS:
        runs[strategy] = {}

        for seed in SEEDS:
            experiment = experiment_id(strategy, seed)
            path = metric_path(experiment)
            predictions = prediction_path(experiment)

            if not path.is_file() or not predictions.is_file():
                raise FileNotFoundError(experiment)

            data = json.loads(path.read_text(encoding="utf-8"))

            if (
                data["experiment_id"] != experiment
                or data["evaluation_split"] != "validation"
                or data["test_evaluated"] is not False
                or data["optimizer_steps"] != 5408
            ):
                raise ValueError(f"Invalid run metadata: {experiment}")

            for metric in METRICS:
                value = float(data[metric])
                if not 0 <= value <= 1:
                    raise ValueError(f"Invalid {metric}: {experiment}")

            runs[strategy][str(seed)] = {
                "experiment_id": experiment,
                "metrics": {metric: float(data[metric]) for metric in METRICS},
                "selected_threshold": float(data["selected_threshold"]),
                "optimizer_steps": 5408,
                "metrics_sha256": sha256(path),
                "predictions_sha256": sha256(predictions),
            }

        summary[strategy] = {}

        for metric in METRICS:
            values = [runs[strategy][str(seed)]["metrics"][metric] for seed in SEEDS]

            summary[strategy][metric] = summarize(values)

    for strategy in EXPERIMENTS:
        if strategy == BASELINE:
            continue

        paired[strategy] = {}

        for reference in (BASELINE, "rare4x"):
            if strategy == reference:
                continue

            paired[strategy][f"vs_{reference}"] = {}

            for metric in METRICS:
                differences = [
                    (
                        runs[strategy][str(seed)]["metrics"][metric]
                        - runs[reference][str(seed)]["metrics"][metric]
                    )
                    for seed in SEEDS
                ]

                paired[strategy][f"vs_{reference}"][metric] = summarize(differences)

    report = {
        "report_id": "multiseed_robustness_v001",
        "dataset": "goemotions_v001",
        "evaluation_split": "validation",
        "test_evaluated": False,
        "seeds": list(SEEDS),
        "strategies": list(EXPERIMENTS),
        "runs": runs,
        "summary": summary,
        "paired_differences": paired,
        "limitations": [
            "Only three paired training seeds.",
            "Thresholds selected on validation.",
            "Synthetic labels were LLM-generated under a one-time review exemption.",
            "Treatment development used validation "
            "results; this is exploratory evidence.",
            "Statistical significance is not established.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(
            report,
            stream,
            indent=2,
            ensure_ascii=False,
        )
        stream.write("\n")

    print("P11-E6C — MULTI-SEED ROBUSTNESS")
    print("-" * 64)

    for strategy in EXPERIMENTS:
        print(f"\n{strategy.upper()}")

        for metric in METRICS:
            item = summary[strategy][metric]
            print(f"  {metric:<24} {item['mean']:.5f} +/- {item['sample_std']:.5f}")

    print("\nMACRO F1 — PAIRED DIFFERENCES")

    for strategy, comparisons in paired.items():
        for reference, metrics in comparisons.items():
            item = metrics["macro_f1"]

            print(f"  {strategy:<12} {reference:<14} {item['mean']:+.5f}")

    print(f"\n[OK] Report saved: {OUTPUT}")
    print("[OK] 12 experiments consolidated")


if __name__ == "__main__":
    main()
