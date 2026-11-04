import hashlib
import json
from pathlib import Path

from ml.src.data.goemotions_dataset import (
    load_goemotions_rows,
)
from ml.src.data.goemotions_sampling import (
    BOOST_FACTOR,
    TARGET_LABELS,
    build_sampling_weights,
    sample_epoch_indices,
    summarize_sampling,
)


ROOT = Path(__file__).resolve().parents[2]

OUTPUT_PATH = ROOT / "reports/goemotions" / "real_sampling_audit_v001.json"

MANIFEST_PATH = ROOT / "data/processed/goemotions_v001/manifest.json"

SEED = 42
EPOCHS = 4


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Audit already exists: {OUTPUT_PATH}")

    rows, label_names = load_goemotions_rows("train")

    if len(rows) != 43252:
        raise ValueError("Unexpected training size")

    report = {
        "audit_id": "real_sampling_audit_v001",
        "dataset": "goemotions_v001",
        "dataset_manifest_sha256": sha256_file(MANIFEST_PATH),
        "seed": SEED,
        "epochs": EPOCHS,
        "sampling_replacement": True,
        "draws_per_epoch": len(rows),
        "boost_factor": BOOST_FACTOR,
        "target_labels": list(TARGET_LABELS),
        "strategies": {},
    }

    for strategy in ("uniform", "rare_oversample"):
        weights = build_sampling_weights(
            rows,
            label_names,
            strategy,
        )

        epoch_reports = []

        print(f"\nStrategy: {strategy}")

        for epoch in range(EPOCHS):
            indices = sample_epoch_indices(
                weights,
                seed=SEED,
                epoch=epoch,
            )

            summary = summarize_sampling(
                rows,
                label_names,
                weights,
                indices,
            )

            summary["epoch"] = epoch + 1
            epoch_reports.append(summary)

            print(
                f"  Epoch {epoch + 1}: "
                f"{summary['epoch_draws']} draws, "
                f"{summary['unique_examples_sampled']} "
                "unique examples"
            )

            for label, info in summary["target_labels"].items():
                ratio = info["exposure_ratio"]

                print(
                    f"    {label:<13} "
                    f"{info['original_positive_occurrences']:>5} "
                    f"-> "
                    f"{info['sampled_positive_occurrences']:>5} "
                    f"({ratio:.2f}x)"
                )

        report["strategies"][strategy] = {
            "epoch_reports": epoch_reports,
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

    print(f"\n[OK] Audit saved: {OUTPUT_PATH}")
    print("[OK] Sampling policies verified")


if __name__ == "__main__":
    main()
