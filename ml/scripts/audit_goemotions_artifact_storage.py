import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = ROOT / "ml/artifacts/models"
METRICS_DIR = ROOT / "ml/artifacts/metrics"
PREDICTIONS_DIR = ROOT / "ml/artifacts/predictions"
LOGS_DIR = ROOT / "ml/artifacts/logs"

OUTPUT = ROOT / "reports/goemotions/artifact_storage_inventory_v001.json"

EXPERIMENTS = {
    "uniform": "roberta_goemotions_uniform_v001",
    "rare4x": "roberta_goemotions_rare4x_v001",
    "synthetic": "roberta_goemotions_synthetic_v002",
    "hybrid": "roberta_goemotions_hybrid_v001",
}

SEEDS = (42, 123, 456)

SELECTED_MODEL = "roberta_goemotions_rare4x_v001"

# Proposed representative models.
# No files will be deleted.
KEEP_MODELS = {
    "roberta_goemotions_uniform_v001",
    "roberta_goemotions_rare4x_v001",
    "roberta_goemotions_synthetic_v002",
    "roberta_goemotions_hybrid_v001",
}


def size_bytes(path):
    if not path.exists():
        return 0

    if path.is_file():
        return path.stat().st_size

    return sum(
        file.stat().st_size
        for file in path.rglob("*")
        if file.is_file() and not file.is_symlink()
    )


def gib(size):
    return size / (1024**3)


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def experiment_id(strategy, seed):
    base = EXPERIMENTS[strategy]

    if seed == 42:
        return base

    return f"{base}_seed{seed}"


def inspect_run(strategy, seed):
    experiment = experiment_id(strategy, seed)

    run_dir = MODELS_DIR / experiment
    checkpoints_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"

    provenance = run_dir / "experiment_provenance.json"

    metrics = METRICS_DIR / f"{experiment}_validation.json"

    predictions = PREDICTIONS_DIR / f"{experiment}_validation.jsonl"

    log = LOGS_DIR / f"{experiment}_training.json"

    required = {
        "run_directory": run_dir,
        "best_model": best_model_dir,
        "provenance": provenance,
        "metrics": metrics,
        "predictions": predictions,
        "training_log": log,
    }

    missing = [name for name, path in required.items() if not path.exists()]

    checkpoint_entries = []

    if checkpoints_dir.is_dir():
        for child in sorted(checkpoints_dir.iterdir()):
            checkpoint_entries.append(
                {
                    "name": child.name,
                    "bytes": size_bytes(child),
                    "gib": round(gib(size_bytes(child)), 4),
                }
            )

    best_model_files = []

    if best_model_dir.is_dir():
        for path in sorted(best_model_dir.iterdir()):
            if path.is_file():
                best_model_files.append(
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                    }
                )

    artifacts = {
        "checkpoints": checkpoints_dir,
        "best_model": best_model_dir,
        "provenance": provenance,
        "metrics": metrics,
        "predictions": predictions,
        "training_log": log,
    }

    sizes = {name: size_bytes(path) for name, path in artifacts.items()}

    return {
        "experiment_id": experiment,
        "strategy": strategy,
        "seed": seed,
        "selected_for_deployment": (experiment == SELECTED_MODEL),
        "proposed_model_retention": (experiment in KEEP_MODELS),
        "missing_artifacts": missing,
        "sizes_bytes": sizes,
        "run_directory_bytes": size_bytes(run_dir),
        "checkpoint_entries": checkpoint_entries,
        "best_model_files": best_model_files,
        "validation_macro_f1": (
            json.loads(metrics.read_text(encoding="utf-8"))["macro_f1"]
            if metrics.is_file()
            else None
        ),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Inventory already exists: {OUTPUT}")

    runs = [inspect_run(strategy, seed) for strategy in EXPERIMENTS for seed in SEEDS]

    missing = [run["experiment_id"] for run in runs if run["missing_artifacts"]]

    total_model_storage = sum(run["run_directory_bytes"] for run in runs)

    checkpoint_storage = sum(run["sizes_bytes"]["checkpoints"] for run in runs)

    best_model_storage = sum(run["sizes_bytes"]["best_model"] for run in runs)

    redundant_model_storage = sum(
        run["sizes_bytes"]["best_model"]
        for run in runs
        if not run["proposed_model_retention"]
    )

    retained_model_storage = sum(
        run["sizes_bytes"]["best_model"]
        for run in runs
        if run["proposed_model_retention"]
    )

    report = {
        "audit_id": "artifact_storage_inventory_v001",
        "read_only": True,
        "model_deletion_performed": False,
        "selected_model": SELECTED_MODEL,
        "proposed_retained_models": sorted(KEEP_MODELS),
        "experiments": runs,
        "summary": {
            "experiment_count": len(runs),
            "total_model_storage_bytes": (total_model_storage),
            "checkpoint_storage_bytes": (checkpoint_storage),
            "best_model_storage_bytes": (best_model_storage),
            "redundant_best_model_bytes": (redundant_model_storage),
            "retained_best_model_bytes": (retained_model_storage),
            "estimated_reclaimable_bytes": (
                checkpoint_storage + redundant_model_storage
            ),
            "incomplete_experiments": missing,
        },
        "limitations": [
            "Inventory covers the 12 controlled runs.",
            "Disk usage is measured using logical file sizes.",
            "Reclaimable bytes are estimates, not guaranteed free-space increases.",
            "No files were modified or deleted.",
            "Model retention is provisional until "
            "inference and provenance verification.",
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

    print("P11-G1 — ARTIFACT STORAGE INVENTORY")
    print("-" * 76)

    print(f"{'Experiment':<49}{'Checkpoints':>13}{'Best Model':>13}")

    for run in sorted(
        runs,
        key=lambda row: row["run_directory_bytes"],
        reverse=True,
    ):
        sizes = run["sizes_bytes"]

        print(
            f"{run['experiment_id']:<49}"
            f"{gib(sizes['checkpoints']):>13.2f}"
            f"{gib(sizes['best_model']):>13.2f}"
        )

    summary = report["summary"]

    print("\nSTORAGE SUMMARY")
    print(
        "Total model directories:",
        f"{gib(total_model_storage):.2f} GiB",
    )
    print(
        "Intermediate checkpoints:",
        f"{gib(checkpoint_storage):.2f} GiB",
    )
    print(
        "Inference-ready models:",
        f"{gib(best_model_storage):.2f} GiB",
    )
    print(
        "Proposed retained models:",
        f"{gib(retained_model_storage):.2f} GiB",
    )
    print(
        "Estimated reclaimable:",
        f"{gib(summary['estimated_reclaimable_bytes']):.2f} GiB",
    )
    print(
        "Incomplete experiments:",
        len(missing),
    )
    print("\n[OK] Inventory saved:", OUTPUT)
    print("[OK] No artifacts deleted")


if __name__ == "__main__":
    main()
