import hashlib
import json
from pathlib import Path

from ml.src.data.goemotions_dataset import (
    load_goemotions_rows,
)
from ml.src.data.goemotions_experiment_sampling import (
    REAL_DRAWS,
    make_epoch_schedule,
    summarize_epoch,
)
from ml.scripts.audit_goemotions_experiment_sampling import (
    load_synthetic,
)


ROOT = Path(__file__).resolve().parents[2]

PREVIOUS_AUDIT = ROOT / "reports/goemotions/controlled_sampling_audit_v001.json"

OUTPUT = ROOT / "reports/goemotions/hybrid_sampling_audit_v001.json"

SEED = 42
EPOCHS = 4

TARGET_LABELS = (
    "grief",
    "nervousness",
    "pride",
    "relief",
)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Audit already exists: {OUTPUT}")

    previous = json.loads(PREVIOUS_AUDIT.read_text(encoding="utf-8"))

    if (
        previous["experiment"] != "controlled_sampling_v001"
        or previous["seed"] != SEED
        or previous["epochs"] != EPOCHS
    ):
        raise ValueError("Previous audit mismatch")

    real_rows, label_names = load_goemotions_rows("train")
    synthetic_rows, synthetic_manifest_path = load_synthetic()

    if sha256_file(synthetic_manifest_path) != previous["synthetic_manifest_sha256"]:
        raise ValueError("Synthetic data version changed")

    report = {
        "experiment": "hybrid_sampling_v001",
        "strategy": "hybrid",
        "seed": SEED,
        "epochs": EPOCHS,
        "draws_per_epoch": REAL_DRAWS,
        "real_oversampling_fraction": 0.5,
        "synthetic_policy_fraction": 0.5,
        "previous_audit_sha256": sha256_file(PREVIOUS_AUDIT),
        "synthetic_manifest_sha256": (sha256_file(synthetic_manifest_path)),
        "epoch_reports": [],
    }

    print("P11-E5 — HYBRID SAMPLING AUDIT")
    print("-" * 55)

    for epoch in range(EPOCHS):
        first = make_epoch_schedule(
            real_rows,
            synthetic_rows,
            label_names,
            "hybrid",
            SEED,
            epoch,
        )

        repeated = make_epoch_schedule(
            real_rows,
            synthetic_rows,
            label_names,
            "hybrid",
            SEED,
            epoch,
        )

        if first != repeated:
            raise ValueError(f"Non-deterministic schedule: epoch {epoch}")

        if len(first) != REAL_DRAWS:
            raise ValueError("Incorrect epoch length")

        summary = summarize_epoch(
            first,
            real_rows,
            synthetic_rows,
            label_names,
        )

        source_counts = summary["source_draws"]

        if sum(source_counts.values()) != REAL_DRAWS:
            raise ValueError("Source count mismatch")

        # Sanity check, not a requirement that the
        # stochastic mixture contain exactly 758
        # synthetic draws.
        synthetic_count = source_counts.get("synthetic", 0)

        if not 500 <= synthetic_count <= 1000:
            raise ValueError(f"Unexpected synthetic exposure: {synthetic_count}")

        report["epoch_reports"].append(summary)

        print(f"\nEpoch {epoch + 1}")
        print("  Total draws:", summary["total_draws"])
        print("  Unique examples:", summary["unique_examples"])
        print("  Real draws:", source_counts.get("real", 0))
        print("  Synthetic draws:", synthetic_count)

        for label in TARGET_LABELS:
            count = summary["label_occurrences"][label]

            reference = previous["strategies"]["rare_oversample"][epoch][
                "label_occurrences"
            ][label]

            delta = count - reference

            print(f"  {label:<13} {count:>5} (vs real4x: {delta:+d})")

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(
            report,
            stream,
            indent=2,
            ensure_ascii=False,
        )
        stream.write("\n")

    print(f"\n[OK] Hybrid audit saved: {OUTPUT}")
    print("[OK] Hybrid schedules verified")


if __name__ == "__main__":
    main()
