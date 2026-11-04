import hashlib
import json
from pathlib import Path

from ml.src.data.goemotions_dataset import (
    load_goemotions_rows,
)
from ml.src.data.goemotions_experiment_sampling import (
    REAL_DRAWS,
    SYNTHETIC_QUOTAS,
    make_epoch_schedule,
    summarize_epoch,
)
from ml.scripts.audit_goemotions_experiment_sampling import (
    load_synthetic,
)


ROOT = Path(__file__).resolve().parents[2]

OUTPUT = ROOT / "reports/goemotions/multiseed_sampling_audit_v001.json"

SEEDS = (123, 456)
EPOCHS = 4

STRATEGIES = (
    "uniform",
    "rare_oversample",
    "synthetic",
    "hybrid",
)

TARGET_LABELS = tuple(SYNTHETIC_QUOTAS)


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def schedule_digest(schedule):
    payload = json.dumps(
        schedule,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    return hashlib.sha256(payload).hexdigest()


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Audit already exists: {OUTPUT}")

    real_rows, label_names = load_goemotions_rows("train")
    synthetic_rows, synthetic_manifest_path = load_synthetic()

    if len(real_rows) != REAL_DRAWS:
        raise ValueError("Unexpected real-data size")

    previous_audit_path = (
        ROOT / "reports/goemotions/controlled_sampling_audit_v001.json"
    )

    hybrid_audit_path = ROOT / "reports/goemotions/hybrid_sampling_audit_v001.json"

    report = {
        "audit_id": "multiseed_sampling_v001",
        "seeds": list(SEEDS),
        "epochs": EPOCHS,
        "strategies": list(STRATEGIES),
        "draws_per_epoch": REAL_DRAWS,
        "source_hashes": {
            "synthetic_manifest": sha256_file(synthetic_manifest_path),
            "controlled_sampling_audit": sha256_file(previous_audit_path),
            "hybrid_sampling_audit": sha256_file(hybrid_audit_path),
        },
        "runs": {},
    }

    print("P11-E6A — MULTI-SEED SAMPLING AUDIT")
    print("-" * 56)

    for seed in SEEDS:
        for strategy in STRATEGIES:
            run_key = f"{strategy}_seed{seed}"
            print(f"\n{run_key}")

            epoch_reports = []

            for epoch in range(EPOCHS):
                schedule = make_epoch_schedule(
                    real_rows,
                    synthetic_rows,
                    label_names,
                    strategy,
                    seed,
                    epoch,
                )

                repeated = make_epoch_schedule(
                    real_rows,
                    synthetic_rows,
                    label_names,
                    strategy,
                    seed,
                    epoch,
                )

                if schedule != repeated:
                    raise ValueError(f"Non-deterministic: {run_key}, epoch {epoch}")

                if len(schedule) != REAL_DRAWS:
                    raise ValueError(f"Incorrect epoch size: {run_key}")

                summary = summarize_epoch(
                    schedule,
                    real_rows,
                    synthetic_rows,
                    label_names,
                )

                source_counts = summary["source_draws"]
                synthetic_draws = source_counts.get("synthetic", 0)

                if strategy == "synthetic":
                    if (
                        synthetic_draws != 1515
                        or summary["synthetic_primary_draws"] != SYNTHETIC_QUOTAS
                    ):
                        raise ValueError(f"Synthetic quota mismatch: {run_key}")

                elif strategy == "hybrid":
                    if not 500 <= synthetic_draws <= 1000:
                        raise ValueError(f"Hybrid exposure unexpected: {run_key}")

                elif synthetic_draws != 0:
                    raise ValueError(f"Synthetic draws in control: {run_key}")

                epoch_reports.append(
                    {
                        **summary,
                        "schedule_sha256": schedule_digest(schedule),
                    }
                )

                labels = summary["label_occurrences"]

                print(
                    f"  Epoch {epoch + 1}: "
                    f"{summary['total_draws']} draws, "
                    f"{summary['unique_examples']} unique, "
                    f"{synthetic_draws} synthetic"
                )

                print(
                    "    "
                    + ", ".join(f"{label}={labels[label]}" for label in TARGET_LABELS)
                )

            report["runs"][run_key] = epoch_reports

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

    print(f"\n[OK] Audit saved: {OUTPUT}")
    print("[OK] Eight new schedules verified")


if __name__ == "__main__":
    main()
