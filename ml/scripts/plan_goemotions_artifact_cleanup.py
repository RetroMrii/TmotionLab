import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "ml/artifacts/models"
REPORTS_DIR = ROOT / "reports/goemotions"

INVENTORY = REPORTS_DIR / "artifact_storage_inventory_v001.json"
VERIFICATION = REPORTS_DIR / "retained_model_verification_v001.json"
FREEZE = REPORTS_DIR / "final_model_freeze_v001.json"
OUTPUT = REPORTS_DIR / "artifact_cleanup_plan_v001.json"

EXPERIMENTS = {
    "uniform": "roberta_goemotions_uniform_v001",
    "rare4x": "roberta_goemotions_rare4x_v001",
    "synthetic": "roberta_goemotions_synthetic_v002",
    "hybrid": "roberta_goemotions_hybrid_v001",
}


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path):
    if path.is_symlink():
        raise ValueError(f"Symlink not permitted: {path}")

    total = 0

    for item in path.rglob("*"):
        if item.is_symlink():
            raise ValueError(f"Symlink not permitted: {item}")
        if item.is_file():
            total += item.stat().st_size

    return total


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Cleanup plan already exists: {OUTPUT}")

    inventory = load_json(INVENTORY)
    verification = load_json(VERIFICATION)
    freeze = load_json(FREEZE)

    if (
        inventory["audit_id"] != "artifact_storage_inventory_v001"
        or inventory["summary"]["experiment_count"] != 12
        or inventory["summary"]["incomplete_experiments"]
    ):
        raise ValueError("Storage inventory incomplete")

    if (
        verification["verification_id"] != "retained_model_verification_v001"
        or verification["frozen_model_verified"] is not True
        or verification["validation_inference_verified"] is not True
        or len(verification["models"]) != 4
    ):
        raise ValueError("Retained models not verified")

    selected = EXPERIMENTS["rare4x"]

    if freeze["primary_experiment"] != selected:
        raise ValueError("Final test model mismatch")

    for strategy, experiment in EXPERIMENTS.items():
        model_dir = MODELS_DIR / experiment / "best_model"

        verified = verification["models"][strategy]

        if verified["experiment_id"] != experiment or verified["verified"] is not True:
            raise ValueError(f"Retained model not verified: {experiment}")

        actual_hashes = {
            path.name: sha256(path)
            for path in model_dir.iterdir()
            if path.is_file() and not path.is_symlink()
        }

        if actual_hashes != verified["model_file_hashes"]:
            raise ValueError(f"Retained model changed: {experiment}")

        if strategy == "rare4x" and actual_hashes != freeze["model_file_hashes"]:
            raise ValueError("Frozen model changed")

    inventory_runs = {run["experiment_id"]: run for run in inventory["experiments"]}

    targets = []

    for strategy, base in EXPERIMENTS.items():
        for seed in (42, 123, 456):
            experiment = base if seed == 42 else f"{base}_seed{seed}"

            if experiment not in inventory_runs:
                raise ValueError(f"Missing inventory: {experiment}")

            run_dir = MODELS_DIR / experiment

            required = (
                run_dir / "experiment_provenance.json",
                ROOT / "ml/artifacts/metrics" / f"{experiment}_validation.json",
                ROOT / "ml/artifacts/predictions" / f"{experiment}_validation.jsonl",
                ROOT / "ml/artifacts/logs" / f"{experiment}_training.json",
            )

            for path in required:
                if not path.is_file():
                    raise FileNotFoundError(path)

            candidates = [run_dir / "checkpoints"]

            if seed != 42:
                candidates.append(run_dir / "best_model")

            for path in candidates:
                if not path.is_dir() or path.is_symlink():
                    raise ValueError(f"Invalid deletion target: {path}")

                # Explicitly constrain targets to the
                # expected experiment directory.
                if path.parent != run_dir:
                    raise ValueError("Unexpected target parent")

                size = directory_size(path)

                category = "checkpoints" if path.name == "checkpoints" else "best_model"

                recorded = inventory_runs[experiment]["sizes_bytes"][category]

                if size != recorded:
                    raise ValueError(f"Storage inventory mismatch: {path}")

                targets.append(
                    {
                        "path": str(path.relative_to(ROOT)),
                        "experiment": experiment,
                        "category": category,
                        "bytes": size,
                    }
                )

    if len(targets) != 20:
        raise ValueError("Expected exactly 20 targets")

    total_bytes = sum(target["bytes"] for target in targets)

    plan = {
        "plan_id": "artifact_cleanup_plan_v001",
        "execution_authorized": False,
        "deletion_performed": False,
        "targets": targets,
        "target_count": len(targets),
        "estimated_reclaimable_bytes": total_bytes,
        "retained_experiments": list(EXPERIMENTS.values()),
        "preserve": [
            "Four seed-42 best_model directories",
            "All experiment provenance files",
            "All 12 validation metrics",
            "All 12 validation prediction files",
            "All 12 training logs",
            "Final held-out evaluation artifacts",
            "All datasets and research reports",
            "Archived six-emotion project artifacts",
        ],
        "source_sha256": {
            "inventory": sha256(INVENTORY),
            "verification": sha256(VERIFICATION),
            "freeze": sha256(FREEZE),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print("P11-G3 — CLEANUP DRY RUN")
    print("-" * 60)

    for target in targets:
        gib = target["bytes"] / (1024**3)
        print(f"{gib:6.2f} GiB  {target['path']}")

    print("\nTargets:", len(targets))
    print("Estimated recovery:", f"{total_bytes / (1024**3):.2f} GiB")
    print("Files deleted: 0")
    print("[OK] Plan saved:", OUTPUT)


if __name__ == "__main__":
    main()
