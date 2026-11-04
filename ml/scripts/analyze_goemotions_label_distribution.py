import hashlib
import json
from collections import Counter
from pathlib import Path

from ml.src.data.goemotions_dataset import (
    load_goemotions_rows,
)


ROOT = Path(__file__).resolve().parents[2]

MANIFEST_PATH = ROOT / "data/processed/goemotions_v001/manifest.json"

OUTPUT_PATH = (
    ROOT / "reports/goemotions" / "label_distribution_and_synthetic_budget_v001.json"
)

TARGET_LABELS = (
    "grief",
    "nervousness",
    "pride",
    "relief",
)

PILOT_TOTAL = 300
SYNTHETIC_MULTIPLIER = 3


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def analyze_distribution(rows, label_names):
    counts = Counter()

    for row in rows:
        labels = row["labels"]

        if len(labels) != len(set(labels)):
            raise ValueError("Duplicate labels in example")

        if any(
            type(label) is not int or not 0 <= label < len(label_names)
            for label in labels
        ):
            raise ValueError("Invalid label ID")

        counts.update(labels)

    total_examples = len(rows)
    total_occurrences = sum(counts.values())

    distribution = []

    for index, name in enumerate(label_names):
        count = counts[index]

        distribution.append(
            {
                "label_id": index,
                "label": name,
                "positive_examples": count,
                "example_prevalence": (count / total_examples),
                "occurrence_share": (count / total_occurrences),
            }
        )

    distribution.sort(
        key=lambda item: (
            -item["positive_examples"],
            item["label"],
        )
    )

    return {
        "training_examples": total_examples,
        "total_positive_label_occurrences": total_occurrences,
        "mean_labels_per_example": (total_occurrences / total_examples),
        "distribution": distribution,
    }


def calculate_budget(
    distribution,
    target_labels=TARGET_LABELS,
    multiplier=SYNTHETIC_MULTIPLIER,
):
    by_name = {item["label"]: item for item in distribution}

    missing = set(target_labels) - set(by_name)

    if missing:
        raise ValueError(f"Missing target labels: {sorted(missing)}")

    if multiplier < 0:
        raise ValueError("Multiplier must be nonnegative")

    most_common = max(item["positive_examples"] for item in distribution)

    result = {}

    for label in target_labels:
        original = by_name[label]["positive_examples"]

        proposed_additions = original * multiplier

        result[label] = {
            "real_positive_occurrences": original,
            "pilot_target": PILOT_TOTAL // len(target_labels),
            "moderate_synthetic_target": (proposed_additions),
            "moderate_combined_target": (original + proposed_additions),
            "synthetic_share_of_moderate_target": (
                proposed_additions / (original + proposed_additions)
            ),
            "additions_to_match_largest_label": max(
                0,
                most_common - original,
            ),
        }

    return {
        "pilot_total": PILOT_TOTAL,
        "moderate_synthetic_multiplier": multiplier,
        "largest_label_count": most_common,
        "target_budgets": result,
        "moderate_total_additions": sum(
            item["moderate_synthetic_target"] for item in result.values()
        ),
        "full_28_label_equalization_estimate": sum(
            most_common - item["positive_examples"] for item in distribution
        ),
    }


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Report already exists: {OUTPUT_PATH}")

    rows, label_names = load_goemotions_rows("train")

    if len(rows) != 43252 or len(label_names) != 28:
        raise ValueError("Unexpected GoEmotions dataset")

    summary = analyze_distribution(
        rows,
        label_names,
    )

    budgets = calculate_budget(summary["distribution"])

    report = {
        "analysis_id": ("label_distribution_and_synthetic_budget_v001"),
        "dataset_version": "goemotions_v001",
        "split": "train",
        "dataset_manifest_sha256": sha256_file(MANIFEST_PATH),
        "summary": summary,
        "synthetic_budget_scenarios": budgets,
        "assumptions": [
            "Counts represent positive-label occurrences.",
            "Budget estimates assume one targeted positive "
            "label occurrence per generated example.",
            "Generated examples may contain additional co-occurring emotion labels.",
            "Generation targets are not approved examples.",
            "No validation or test text was used.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nGOEMOTIONS TRAINING DISTRIBUTION")
    print("-" * 59)

    print(f"Training examples: {summary['training_examples']}")

    print(f"Positive-label occurrences: {summary['total_positive_label_occurrences']}")

    print(f"Mean labels per example: {summary['mean_labels_per_example']:.3f}")

    print("\n28 labels (most to least frequent):")
    print("-" * 59)

    for item in summary["distribution"]:
        print(
            f"{item['label']:<18} "
            f"{item['positive_examples']:>6} "
            f"{item['example_prevalence'] * 100:>7.2f}%"
        )

    print("\nSYNTHETIC BUDGET SCENARIOS")
    print("-" * 59)

    print(f"Largest label count: {budgets['largest_label_count']}")

    print(
        "Estimated additions for complete 28-label "
        "equalization: "
        f"{budgets['full_28_label_equalization_estimate']}"
    )

    print(f"\nPilot target: {budgets['pilot_total']} examples")

    for label, item in budgets["target_budgets"].items():
        print(
            f"{label:<13} "
            f"real={item['real_positive_occurrences']:<5} "
            f"pilot={item['pilot_target']:<4} "
            f"moderate_add={item['moderate_synthetic_target']:<5} "
            f"full_balance_add="
            f"{item['additions_to_match_largest_label']}"
        )

    print(f"\nModerate strategy total additions: {budgets['moderate_total_additions']}")

    print(f"\n[OK] Report saved: {OUTPUT_PATH}")
    print("[OK] P11-E2A distribution analysis complete")


if __name__ == "__main__":
    main()
