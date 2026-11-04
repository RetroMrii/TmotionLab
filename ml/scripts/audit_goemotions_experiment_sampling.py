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


ROOT = Path(__file__).resolve().parents[2]

SYNTHETIC_DIR = ROOT / "data/processed/synthetic_goemotions_v002"

REPORT_PATH = ROOT / "reports/goemotions/controlled_sampling_audit_v001.json"

SEED = 42
EPOCHS = 4


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_synthetic():
    manifest_path = SYNTHETIC_DIR / "manifest.json"
    data_path = SYNTHETIC_DIR / "train.jsonl"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest["dataset_version"] != "synthetic_goemotions_v002":
        raise ValueError("Unexpected synthetic dataset")

    if sha256_file(data_path) != manifest["sha256"]:
        raise ValueError("Synthetic checksum mismatch")

    with data_path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    if len(rows) != manifest["rows"] or len(rows) != 299:
        raise ValueError("Synthetic row count mismatch")

    seen_ids = set()

    for row in rows:
        if row["example_id"] in seen_ids:
            raise ValueError("Duplicate synthetic ID")

        seen_ids.add(row["example_id"])

        if (
            hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            != row["text_sha256"]
        ):
            raise ValueError("Synthetic text checksum mismatch")

        labels = row["labels"]

        if (
            not labels
            or len(labels) != len(set(labels))
            or any(type(x) is not int or not 0 <= x < 28 for x in labels)
        ):
            raise ValueError("Invalid synthetic labels")

    return rows, manifest_path


def main():
    if REPORT_PATH.exists():
        raise FileExistsError(f"Audit already exists: {REPORT_PATH}")

    real_rows, label_names = load_goemotions_rows("train")
    synthetic_rows, synthetic_manifest = load_synthetic()

    label_to_id = {name: i for i, name in enumerate(label_names)}

    for row in synthetic_rows:
        primary = row["intended_primary_label"]

        if primary not in SYNTHETIC_QUOTAS or label_to_id[primary] not in row["labels"]:
            raise ValueError("Synthetic primary-label mismatch")

    report = {
        "experiment": "controlled_sampling_v001",
        "seed": SEED,
        "epochs": EPOCHS,
        "draws_per_epoch": REAL_DRAWS,
        "synthetic_manifest_sha256": sha256_file(synthetic_manifest),
        "synthetic_quotas": SYNTHETIC_QUOTAS,
        "strategies": {},
    }

    for strategy in (
        "uniform",
        "rare_oversample",
        "synthetic",
    ):
        print(f"\nStrategy: {strategy}")
        epoch_reports = []

        for epoch in range(EPOCHS):
            schedule = make_epoch_schedule(
                real_rows,
                synthetic_rows,
                label_names,
                strategy,
                SEED,
                epoch,
            )

            summary = summarize_epoch(
                schedule,
                real_rows,
                synthetic_rows,
                label_names,
            )

            if summary["total_draws"] != REAL_DRAWS:
                raise ValueError("Incorrect draw count")

            if strategy == "synthetic":
                if summary["source_draws"] != {
                    "real": 41737,
                    "synthetic": 1515,
                }:
                    raise ValueError("Incorrect mixture")

                if summary["synthetic_primary_draws"] != SYNTHETIC_QUOTAS:
                    raise ValueError("Incorrect synthetic quotas")

            else:
                if summary["source_draws"] != {"real": REAL_DRAWS}:
                    raise ValueError("Non-real samples in control")

            epoch_reports.append(summary)

            print(
                f"  Epoch {epoch + 1}: "
                f"{summary['total_draws']} draws, "
                f"{summary['unique_examples']} unique"
            )

            for label in SYNTHETIC_QUOTAS:
                print(f"    {label:<13} {summary['label_occurrences'][label]}")

        report["strategies"][strategy] = epoch_reports

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")

    print(f"\n[OK] Audit saved: {REPORT_PATH}")
    print("[OK] Controlled sampling verified")


if __name__ == "__main__":
    main()
