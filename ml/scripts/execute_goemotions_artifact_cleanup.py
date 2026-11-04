import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2].resolve()
MODELS = ROOT / "ml/artifacts/models"
REPORTS = ROOT / "reports/goemotions"

PLAN_PATH = REPORTS / "artifact_cleanup_plan_v001.json"
INVENTORY_PATH = REPORTS / "artifact_storage_inventory_v001.json"
VERIFICATION_PATH = REPORTS / "retained_model_verification_v001.json"
FREEZE_PATH = REPORTS / "final_model_freeze_v001.json"

RECEIPT_PATH = REPORTS / "artifact_cleanup_receipt_v001.json"

EXPERIMENTS = {
    "uniform": "roberta_goemotions_uniform_v001",
    "rare4x": "roberta_goemotions_rare4x_v001",
    "synthetic": "roberta_goemotions_synthetic_v002",
    "hybrid": "roberta_goemotions_hybrid_v001",
}

CONFIRMATION = "DELETE-20-VERIFIED-DIRECTORIES"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_targets():
    targets = {}

    for base in EXPERIMENTS.values():
        for seed in (42, 123, 456):
            experiment = base if seed == 42 else f"{base}_seed{seed}"

            checkpoint = MODELS / experiment / "checkpoints"
            targets[str(checkpoint.relative_to(ROOT))] = checkpoint

            if seed != 42:
                best_model = MODELS / experiment / "best_model"
                targets[str(best_model.relative_to(ROOT))] = best_model

    if len(targets) != 20:
        raise RuntimeError("Invalid deletion allowlist")

    return targets


def directory_size(path):
    if not path.is_dir() or path.is_symlink():
        raise ValueError(f"Invalid directory: {path}")

    total = 0

    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError(f"Symlink detected: {child}")

        if child.is_file():
            total += child.stat().st_size

    return total


def verify_retained_models(verification, freeze):
    if (
        verification["verification_id"] != "retained_model_verification_v001"
        or verification["frozen_model_verified"] is not True
        or verification["validation_inference_verified"] is not True
    ):
        raise ValueError("Invalid verification report")

    for strategy, experiment in EXPERIMENTS.items():
        model_dir = MODELS / experiment / "best_model"

        if not model_dir.is_dir() or model_dir.is_symlink():
            raise ValueError(f"Retained model unavailable: {experiment}")

        expected = verification["models"][strategy]["model_file_hashes"]

        actual = {}

        for child in model_dir.iterdir():
            if child.is_symlink():
                raise ValueError(f"Symlink in retained model: {child}")
            if not child.is_file():
                raise ValueError(f"Unexpected retained-model entry: {child}")
            actual[child.name] = sha256(child)

        if actual != expected:
            raise ValueError(f"Retained model changed: {experiment}")

        if strategy == "rare4x":
            if (
                experiment != freeze["primary_experiment"]
                or actual != freeze["model_file_hashes"]
            ):
                raise ValueError("Frozen test model integrity mismatch")


def preflight():
    if RECEIPT_PATH.exists():
        raise FileExistsError(
            "Cleanup receipt already exists. Inspect it before any further action."
        )

    plan = read_json(PLAN_PATH)
    inventory = read_json(INVENTORY_PATH)
    verification = read_json(VERIFICATION_PATH)
    freeze = read_json(FREEZE_PATH)

    if (
        plan["plan_id"] != "artifact_cleanup_plan_v001"
        or plan["target_count"] != 20
        or len(plan["targets"]) != 20
        or plan["deletion_performed"] is not False
        or plan["execution_authorized"] is not False
    ):
        raise ValueError("Unexpected cleanup plan")

    if (
        inventory["audit_id"] != "artifact_storage_inventory_v001"
        or inventory["summary"]["experiment_count"] != 12
        or inventory["summary"]["incomplete_experiments"]
    ):
        raise ValueError("Invalid storage inventory")

    source_paths = {
        "inventory": INVENTORY_PATH,
        "verification": VERIFICATION_PATH,
        "freeze": FREEZE_PATH,
    }

    for name, path in source_paths.items():
        if sha256(path) != plan["source_sha256"][name]:
            raise ValueError(f"Cleanup source changed: {name}")

    verify_retained_models(verification, freeze)

    allowed = expected_targets()
    seen = set()
    checked = []

    for target in plan["targets"]:
        raw = target["path"]
        relative = Path(raw)

        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe target path: {raw}")

        normalized = str(relative)

        if normalized not in allowed:
            raise ValueError(f"Target not allowlisted: {raw}")

        if normalized in seen:
            raise ValueError(f"Repeated deletion target: {raw}")

        seen.add(normalized)
        path = allowed[normalized]

        # Do not follow links in the expected parent chain.
        parent = path.parent

        if parent.is_symlink() or path.is_symlink():
            raise ValueError(f"Linked deletion path: {path}")

        if path.resolve() != path:
            raise ValueError(f"Unexpected resolved path: {path}")

        size = directory_size(path)

        if size != target["bytes"]:
            raise ValueError(f"Directory changed since dry run: {path}")

        expected_category = (
            "checkpoints" if path.name == "checkpoints" else "best_model"
        )

        if target["category"] != expected_category:
            raise ValueError(f"Target category mismatch: {path}")

        if target["experiment"] != path.parent.name:
            raise ValueError(f"Experiment mismatch: {path}")

        checked.append(
            {
                "path": normalized,
                "bytes": size,
            }
        )

    if seen != set(allowed):
        raise ValueError("Deletion allowlist mismatch")

    total = sum(item["bytes"] for item in checked)

    if total != plan["estimated_reclaimable_bytes"]:
        raise ValueError("Recovery estimate mismatch")

    return checked, total


def write_receipt(receipt):
    RECEIPT_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary = RECEIPT_PATH.with_suffix(".tmp")

    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            receipt,
            stream,
            indent=2,
            ensure_ascii=False,
        )
        stream.write("\n")

    temporary.replace(RECEIPT_PATH)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--execute",
        action="store_true",
    )

    parser.add_argument(
        "--confirm",
        default="",
    )

    args = parser.parse_args()

    checked, total = preflight()

    print("P11-G3 — VERIFIED CLEANUP")
    print("-" * 55)
    print("Verified targets:", len(checked))
    print("Estimated recovery:", f"{total / (1024**3):.2f} GiB")
    print("Retained models: 4, hash-verified")

    if not args.execute:
        print("[DRY RUN] No files deleted")
        return

    if args.confirm != CONFIRMATION:
        raise ValueError("Explicit deletion confirmation required")

    receipt = {
        "cleanup_id": "artifact_cleanup_v001",
        "plan_sha256": sha256(PLAN_PATH),
        "status": "in_progress",
        "completed_targets": [],
        "total_deleted_bytes": 0,
    }

    # Create a durable receipt before the first deletion.
    write_receipt(receipt)

    for item in checked:
        path = ROOT / item["path"]

        print(f"[DELETE] {item['path']}", flush=True)

        try:
            # Recheck each target immediately before removal.
            if directory_size(path) != item["bytes"]:
                raise ValueError(f"Target changed during cleanup: {path}")

            shutil.rmtree(path)

        except Exception:
            receipt["status"] = "interrupted"
            write_receipt(receipt)
            raise

        receipt["completed_targets"].append(item)
        receipt["total_deleted_bytes"] += item["bytes"]
        write_receipt(receipt)

    verify_retained_models(
        read_json(VERIFICATION_PATH),
        read_json(FREEZE_PATH),
    )

    receipt["status"] = "complete"
    write_receipt(receipt)

    print("\n[OK] Cleanup completed")
    print(
        "Deleted:",
        len(receipt["completed_targets"]),
        "directories",
    )
    print(
        "Logical storage removed:",
        f"{receipt['total_deleted_bytes'] / (1024**3):.2f} GiB",
    )
    print("[OK] Retained models verified")
    print("[OK] Receipt:", RECEIPT_PATH)


if __name__ == "__main__":
    main()
