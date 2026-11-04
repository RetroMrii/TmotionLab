import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from ml.src.data.goemotions_dataset import (
    load_goemotions_rows,
)


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENTS = {
    "uniform": "roberta_goemotions_uniform_v001",
    "rare4x": "roberta_goemotions_rare4x_v001",
    "synthetic": "roberta_goemotions_synthetic_v002",
    "hybrid": "roberta_goemotions_hybrid_v001",
}

FREEZE_PATH = ROOT / "reports/goemotions/final_model_freeze_v001.json"

OUTPUT = ROOT / "reports/goemotions/retained_model_verification_v001.json"

SAMPLE_SIZE = 16
PROBABILITY_TOLERANCE = 0.002


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_model(strategy, experiment, rows, names, freeze):
    model_dir = ROOT / "ml/artifacts/models" / experiment / "best_model"

    metrics_path = ROOT / "ml/artifacts/metrics" / f"{experiment}_validation.json"

    predictions_path = (
        ROOT / "ml/artifacts/predictions" / f"{experiment}_validation.jsonl"
    )

    metrics = load_json(metrics_path)

    if (
        metrics["experiment_id"] != experiment
        or metrics["evaluation_split"] != "validation"
    ):
        raise ValueError(f"Incorrect metrics: {experiment}")

    actual_hashes = {
        file.name: sha256(file) for file in model_dir.iterdir() if file.is_file()
    }

    if actual_hashes != metrics["model_files"]:
        raise ValueError(f"Model-file hash mismatch: {experiment}")

    if strategy == "rare4x":
        if (
            experiment != freeze["primary_experiment"]
            or actual_hashes != freeze["model_file_hashes"]
        ):
            raise ValueError("Final test model differs from frozen checkpoint")

    config = load_json(model_dir / "config.json")

    id2label = config.get("id2label")

    if not isinstance(id2label, dict):
        raise ValueError(f"Missing label mapping: {experiment}")

    if len(id2label) != 28:
        raise ValueError(f"Incorrect label count: {experiment}")

    for index, name in enumerate(names):
        if id2label.get(str(index)) != name:
            raise ValueError(f"Label mapping mismatch: {experiment}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
    )

    model.eval()
    model.to("cpu")

    inputs = tokenizer(
        [row["text"] for row in rows[:SAMPLE_SIZE]],
        padding=True,
        truncation=True,
        max_length=48,
        return_tensors="pt",
    )

    with torch.inference_mode():
        logits = model(**inputs).logits
        if logits.shape != (SAMPLE_SIZE, 28):
            raise ValueError(
                f"Unexpected model output shape: {experiment}: {tuple(logits.shape)}"
            )
        probabilities = torch.sigmoid(logits).cpu().numpy()

    expected_rows = []

    with predictions_path.open("r", encoding="utf-8") as stream:
        for line in stream:
            expected_rows.append(json.loads(line))

            if len(expected_rows) == SAMPLE_SIZE:
                break

    if len(expected_rows) != SAMPLE_SIZE:
        raise ValueError(f"Insufficient predictions: {experiment}")

    for source, predicted in zip(
        rows[:SAMPLE_SIZE],
        expected_rows,
    ):
        if (
            source["example_id"] != predicted["example_id"]
            or source["source_id"] != predicted["source_id"]
        ):
            raise ValueError(f"Prediction alignment mismatch: {experiment}")

    expected_probabilities = np.asarray(
        [row["probabilities"] for row in expected_rows],
        dtype=np.float32,
    )

    max_difference = float(np.max(np.abs(probabilities - expected_probabilities)))

    if max_difference > PROBABILITY_TOLERANCE:
        raise ValueError(
            f"Inference mismatch: {experiment}, max difference={max_difference:.6f}"
        )

    print(f"[OK] {strategy:<10} max probability difference={max_difference:.6f}")

    return {
        "experiment_id": experiment,
        "verified": True,
        "model_file_hashes": actual_hashes,
        "sample_size": SAMPLE_SIZE,
        "maximum_probability_difference": max_difference,
        "metrics_sha256": sha256(metrics_path),
        "predictions_sha256": sha256(predictions_path),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Verification report already exists: {OUTPUT}")

    rows, names = load_goemotions_rows("validation")

    if len(rows) != 5426 or len(names) != 28:
        raise ValueError("Unexpected validation dataset")

    freeze = load_json(FREEZE_PATH)

    results = {}

    for strategy, experiment in EXPERIMENTS.items():
        results[strategy] = verify_model(strategy, experiment, rows, names, freeze)

    report = {
        "verification_id": ("retained_model_verification_v001"),
        "models": results,
        "frozen_model_verified": True,
        "validation_inference_verified": True,
        "deletion_performed": False,
        "probability_tolerance": PROBABILITY_TOLERANCE,
        "limitations": [
            "Inference comparison covers 16 validation examples per model.",
            "Small numeric differences are permitted across inference environments.",
            "Training-resumption capability is not tested.",
        ],
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print("\n[OK] Four retained models verified")
    print("[OK] Frozen test model unchanged")
    print("[OK] Report saved:", OUTPUT)
    print("[OK] No artifacts deleted")


if __name__ == "__main__":
    main()
