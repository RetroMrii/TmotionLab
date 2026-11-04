import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from ml.src.data.goemotions_dataset import (
    load_goemotions_rows,
)


ROOT = Path(__file__).resolve().parents[2]

EXPERIMENT = "roberta_goemotions_rare4x_v001"
THRESHOLD = 0.25

MODEL_DIR = ROOT / "ml/artifacts/models" / EXPERIMENT / "best_model"

METRICS_PATH = ROOT / "ml/artifacts/metrics" / f"{EXPERIMENT}_validation.json"

PROVENANCE_PATH = (
    ROOT / "ml/artifacts/models" / EXPERIMENT / "experiment_provenance.json"
)

REAL_MANIFEST = ROOT / "data/processed/goemotions_v001/manifest.json"

FREEZE_PATH = ROOT / "reports/goemotions/final_model_freeze_v001.json"

RESULT_PATH = ROOT / "reports/goemotions/final_heldout_test_v001.json"

PREDICTIONS_PATH = ROOT / "reports/goemotions/final_heldout_predictions_v001.jsonl"

TARGET_LABELS = ("grief", "nervousness", "pride", "relief")


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def model_hashes():
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(MODEL_DIR)

    paths = sorted(path for path in MODEL_DIR.iterdir() if path.is_file())

    if not paths:
        raise ValueError("No model files found")

    return {path.name: sha256_file(path) for path in paths}


def freeze():
    if FREEZE_PATH.exists() or RESULT_PATH.exists() or PREDICTIONS_PATH.exists():
        raise FileExistsError("Freeze or test artifacts already exist")

    metrics = load_json(METRICS_PATH)
    provenance = load_json(PROVENANCE_PATH)
    real_manifest = load_json(REAL_MANIFEST)

    if metrics["experiment_id"] != EXPERIMENT:
        raise ValueError("Wrong validation experiment")

    if provenance["experiment_id"] != EXPERIMENT:
        raise ValueError("Wrong model provenance")

    if provenance["strategy"] != "rare_oversample":
        raise ValueError("Wrong sampling strategy")

    if provenance["seed"] != 42:
        raise ValueError("Unexpected training seed")

    if metrics["evaluation_split"] != "validation":
        raise ValueError("Model not selected on validation")

    if metrics["test_evaluated"] is not False:
        raise ValueError("Unexpected test evaluation status")

    if metrics["optimizer_steps"] != 5408:
        raise ValueError("Unexpected training budget")

    if not np.isclose(
        float(metrics["selected_threshold"]),
        THRESHOLD,
        atol=1e-12,
        rtol=0,
    ):
        raise ValueError("Threshold changed")

    if real_manifest.get("dataset_version") != "goemotions_v001":
        raise ValueError("Wrong dataset version")

    if sha256_file(REAL_MANIFEST) != provenance["real_dataset_manifest_sha256"]:
        raise ValueError("Real dataset provenance mismatch")

    hashes = model_hashes()

    if not ("model.safetensors" in hashes or "pytorch_model.bin" in hashes):
        raise ValueError("Model weights missing")

    freeze_record = {
        "freeze_id": "final_model_freeze_v001",
        "primary_experiment": EXPERIMENT,
        "strategy": "rare_oversample",
        "seed": 42,
        "checkpoint_selection": ("highest_validation_macro_average_precision"),
        "classification_threshold": THRESHOLD,
        "threshold_selection_split": "validation",
        "dataset_version": "goemotions_v001",
        "expected_test_examples": 5427,
        "model_dir": str(MODEL_DIR.relative_to(ROOT)),
        "model_file_hashes": hashes,
        "validation_metrics_sha256": sha256_file(METRICS_PATH),
        "provenance_sha256": sha256_file(PROVENANCE_PATH),
        "dataset_manifest_sha256": sha256_file(REAL_MANIFEST),
        "test_evaluated_at_freeze": False,
        "test_time_tuning_allowed": False,
        "research_alternative": ("roberta_goemotions_hybrid_v001"),
        "seed_selection_rule": ("fixed_seed_42_not_best_seed_search"),
    }

    save_json_new(FREEZE_PATH, freeze_record)

    print("[OK] Final model frozen")
    print("Experiment:", EXPERIMENT)
    print("Threshold:", THRESHOLD)
    print("Model files:", len(hashes))
    print("Freeze manifest:", FREEZE_PATH)
    print("Test examples accessed: 0")


def verify_freeze():
    frozen = load_json(FREEZE_PATH)

    if (
        frozen["freeze_id"] != "final_model_freeze_v001"
        or frozen["primary_experiment"] != EXPERIMENT
        or frozen["strategy"] != "rare_oversample"
        or frozen["seed"] != 42
        or frozen["classification_threshold"] != THRESHOLD
        or frozen["dataset_version"] != "goemotions_v001"
        or frozen["test_time_tuning_allowed"] is not False
    ):
        raise ValueError("Frozen specification mismatch")

    if model_hashes() != frozen["model_file_hashes"]:
        raise ValueError("Frozen model files changed")

    checks = (
        (
            METRICS_PATH,
            frozen["validation_metrics_sha256"],
        ),
        (
            PROVENANCE_PATH,
            frozen["provenance_sha256"],
        ),
        (
            REAL_MANIFEST,
            frozen["dataset_manifest_sha256"],
        ),
    )

    for path, expected in checks:
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen source changed: {path}")

    return frozen


@torch.inference_mode()
def predict_probabilities(rows):
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
    )

    if model.config.num_labels != 28:
        raise ValueError("Model output count is not 28")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    model.eval()

    outputs = []
    batch_size = 32

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]

        inputs = tokenizer(
            [row["text"] for row in batch],
            padding=True,
            truncation=True,
            max_length=48,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        logits = model(**inputs).logits

        outputs.append(torch.sigmoid(logits).float().cpu().numpy())

    return np.concatenate(outputs, axis=0)


def evaluate_test():
    if not FREEZE_PATH.exists():
        raise FileNotFoundError("Freeze the final model before evaluation")

    if RESULT_PATH.exists() or PREDICTIONS_PATH.exists():
        raise FileExistsError("Held-out evaluation artifacts already exist")

    frozen = verify_freeze()

    # First access to the test examples in this script.
    rows, label_names = load_goemotions_rows("test")

    if len(rows) != frozen["expected_test_examples"] or len(label_names) != 28:
        raise ValueError("Unexpected test dataset")

    model_config = load_json(MODEL_DIR / "config.json")

    for index, label in enumerate(label_names):
        if model_config["id2label"].get(str(index)) != label:
            raise ValueError(f"Model label mapping mismatch: {index}")

    probabilities = predict_probabilities(rows)

    truth = np.zeros(
        (len(rows), len(label_names)),
        dtype=np.int8,
    )

    for index, row in enumerate(rows):
        truth[index, row["labels"]] = 1

    if probabilities.shape != truth.shape or not np.isfinite(probabilities).all():
        raise ValueError("Invalid prediction shape or values")

    predicted = (probabilities >= THRESHOLD).astype(np.int8)

    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        average=None,
        zero_division=0,
    )

    per_label = {}

    for index, label in enumerate(label_names):
        actual = truth[:, index]
        chosen = predicted[:, index]

        tp = int(np.sum((actual == 1) & (chosen == 1)))
        fp = int(np.sum((actual == 0) & (chosen == 1)))
        fn = int(np.sum((actual == 1) & (chosen == 0)))

        ap = (
            float(average_precision_score(actual, probabilities[:, index]))
            if support[index] > 0
            else None
        )

        per_label[label] = {
            "support": int(support[index]),
            "predicted_positive": int(chosen.sum()),
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "average_precision": ap,
        }

    report = {
        "report_id": "final_heldout_test_v001",
        "freeze_sha256": sha256_file(FREEZE_PATH),
        "experiment_id": EXPERIMENT,
        "split": "test",
        "examples": len(rows),
        "threshold": THRESHOLD,
        "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(truth, predicted, average="micro", zero_division=0)),
        "macro_average_precision": float(
            average_precision_score(truth, probabilities, average="macro")
        ),
        "exact_match_accuracy": float(accuracy_score(truth, predicted)),
        "example_based_f1": float(
            f1_score(truth, predicted, average="samples", zero_division=0)
        ),
        "mean_true_labels": float(truth.sum(axis=1).mean()),
        "mean_predicted_labels": float(predicted.sum(axis=1).mean()),
        "zero_label_predictions": int(np.sum(predicted.sum(axis=1) == 0)),
        "multi_label_predictions": int(np.sum(predicted.sum(axis=1) > 1)),
        "per_label": per_label,
        "target_labels": list(TARGET_LABELS),
        "test_time_tuning_performed": False,
    }

    PREDICTIONS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with PREDICTIONS_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            record = {
                "example_id": row["example_id"],
                "source_id": row["source_id"],
                "true_labels": row["labels"],
                "predicted_labels": np.flatnonzero(predicted[index]).tolist(),
                "probabilities": probabilities[index].tolist(),
            }

            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    report["predictions_sha256"] = sha256_file(PREDICTIONS_PATH)

    save_json_new(RESULT_PATH, report)

    print("P11-F1 — FINAL HELD-OUT TEST")
    print("-" * 52)
    print("Experiment:", EXPERIMENT)
    print("Test examples:", len(rows))
    print("Threshold:", THRESHOLD)

    for metric in (
        "macro_f1",
        "micro_f1",
        "macro_average_precision",
        "exact_match_accuracy",
        "example_based_f1",
    ):
        print(f"{metric}: {report[metric]:.6f}")

    print("\nRare-label F1")

    for label in TARGET_LABELS:
        item = per_label[label]
        print(f"  {label:<13} F1={item['f1']:.4f} support={item['support']}")

    print("[OK] Test evaluation completed")
    print("[OK] Results:", RESULT_PATH)
    print("[OK] Predictions:", PREDICTIONS_PATH)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=("freeze", "evaluate"),
    )
    args = parser.parse_args()

    if args.action == "freeze":
        freeze()
    else:
        evaluate_test()


if __name__ == "__main__":
    main()
