import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ml.src.data.goemotions_dataset import (
    GoEmotionsDataset,
    GoEmotionsCollator,
    load_goemotions_rows,
)
from ml.scripts.train_goemotions_tfidf import (
    choose_threshold,
    evaluate,
)


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "roberta_goemotions_multilabel_real_v001"

CONFIG_PATH = ROOT / "ml/configs" / f"{EXPERIMENT}.json"
DATA_MANIFEST = ROOT / "data/processed/goemotions_v001/manifest.json"

RUN_DIR = ROOT / "ml/artifacts/models" / EXPERIMENT
MODEL_DIR = RUN_DIR / "best_model"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"

METRICS_PATH = ROOT / "ml/artifacts/metrics" / f"{EXPERIMENT}_validation.json"
PREDICTIONS_PATH = ROOT / "ml/artifacts/predictions" / f"{EXPERIMENT}_validation.jsonl"
LOG_PATH = ROOT / "ml/artifacts/logs" / f"{EXPERIMENT}_training.json"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def sigmoid(logits):
    # Numerically stable sigmoid for NumPy arrays.
    logits = np.asarray(logits, dtype=np.float64)
    positive = logits >= 0

    result = np.empty_like(logits)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))

    exp_logits = np.exp(logits[~positive])
    result[~positive] = exp_logits / (1.0 + exp_logits)

    return result


def make_targets(rows, num_labels):
    targets = np.zeros(
        (len(rows), num_labels),
        dtype=np.int8,
    )

    for index, row in enumerate(rows):
        targets[index, row["labels"]] = 1

    return targets


def compute_metrics(eval_prediction):
    logits = eval_prediction.predictions
    labels = eval_prediction.label_ids

    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = sigmoid(logits)
    truth = np.asarray(labels, dtype=np.int8)

    if probabilities.shape != truth.shape:
        raise ValueError("Evaluation shape mismatch")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite evaluation probabilities")

    macro_ap = average_precision_score(
        truth,
        probabilities,
        average="macro",
    )

    return {"macro_average_precision": float(macro_ap)}


def validate_config(config, manifest):
    if config["experiment_id"] != EXPERIMENT:
        raise ValueError("Experiment ID mismatch")

    if config["dataset_version"] != manifest["dataset_version"]:
        raise ValueError("Dataset version mismatch")

    if config["num_labels"] != len(manifest["label_names"]):
        raise ValueError("Label count mismatch")

    if not 2 <= config["max_length"] <= 512:
        raise ValueError("Invalid sequence length")

    if config["checkpoint_selection"]["metric"] != ("macro_average_precision"):
        raise ValueError("Unexpected selection metric")

    candidates = config["threshold_candidates"]

    if (
        not candidates
        or len(set(candidates)) != len(candidates)
        or any(not 0 < value < 1 for value in candidates)
    ):
        raise ValueError("Invalid threshold candidates")


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(DATA_MANIFEST.read_text(encoding="utf-8"))

    validate_config(config, manifest)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Verify GPU-enabled PyTorch.")

    # Protect immutable experiment results.
    if RUN_DIR.exists():
        raise FileExistsError(f"Experiment directory already exists: {RUN_DIR}")

    for path in (
        METRICS_PATH,
        PREDICTIONS_PATH,
        LOG_PATH,
    ):
        if path.exists():
            raise FileExistsError(f"Experiment artifact already exists: {path}")

    set_seed(config["seed"])

    label_names = manifest["label_names"]
    num_labels = config["num_labels"]

    train_rows, train_names = load_goemotions_rows("train")
    val_rows, val_names = load_goemotions_rows("validation")

    if train_names != label_names or val_names != label_names:
        raise ValueError("Dataset label mapping mismatch")

    if len(train_rows) != 43252 or len(val_rows) != 5426:
        raise ValueError("Unexpected split sizes")

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer required")

    train_dataset = GoEmotionsDataset(
        train_rows,
        tokenizer,
        max_length=config["max_length"],
    )
    val_dataset = GoEmotionsDataset(
        val_rows,
        tokenizer,
        max_length=config["max_length"],
    )

    id2label = {index: name for index, name in enumerate(label_names)}
    label2id = {name: index for index, name in enumerate(label_names)}

    model_config = AutoConfig.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        config=model_config,
    )

    if model.config.problem_type != "multi_label_classification":
        raise ValueError("Incorrect classification task")

    train_cfg = config["training"]

    RUN_DIR.mkdir(parents=True, exist_ok=False)

    args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=(train_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=(train_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=(train_cfg["gradient_accumulation_steps"]),
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        warmup_steps=train_cfg["warmup_ratio"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        fp16=train_cfg["fp16"],
        seed=config["seed"],
        data_seed=config["seed"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_average_precision",
        greater_is_better=True,
        save_total_limit=2,
        eval_accumulation_steps=16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=GoEmotionsCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Starting RoBERTa training...")
    train_result = trainer.train()

    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("No best checkpoint selected")

    print(
        "Best checkpoint:",
        trainer.state.best_model_checkpoint,
    )
    print(
        "Best validation macro AP:",
        trainer.state.best_metric,
    )

    # Trainer has restored the selected best checkpoint.
    prediction_output = trainer.predict(val_dataset)

    logits = prediction_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = sigmoid(logits)
    y_true = make_targets(val_rows, num_labels)

    if probabilities.shape != y_true.shape:
        raise ValueError("Final prediction shape mismatch")

    selected_threshold, threshold_results = choose_threshold(
        y_true,
        probabilities,
        config["threshold_candidates"],
    )

    metrics = evaluate(
        y_true,
        probabilities,
        selected_threshold,
        label_names,
    )

    predicted = (probabilities >= selected_threshold).astype(np.int8)

    metrics.update(
        {
            "experiment_id": EXPERIMENT,
            "dataset_version": config["dataset_version"],
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "evaluation_split": "validation",
            "test_evaluated": False,
            "best_checkpoint": (Path(trainer.state.best_model_checkpoint).name),
            "best_checkpoint_macro_ap": float(trainer.state.best_metric),
            "selected_threshold": selected_threshold,
            "threshold_search": threshold_results,
            "config_sha256": sha256_file(CONFIG_PATH),
            "dataset_manifest_sha256": sha256_file(DATA_MANIFEST),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        }
    )

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    with PREDICTIONS_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(val_rows):
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

    metrics["model_files"] = {
        path.name: sha256_file(path) for path in MODEL_DIR.iterdir() if path.is_file()
    }

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    LOG_PATH.write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT,
                "training_metrics": train_result.metrics,
                "best_metric": trainer.state.best_metric,
                "best_checkpoint": (trainer.state.best_model_checkpoint),
                "history": trainer.state.log_history,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Selected threshold: {selected_threshold:.2f}")
    print(f"Validation macro F1: {metrics['macro_f1']:.4f}")
    print(f"Validation micro F1: {metrics['micro_f1']:.4f}")
    print(f"Validation macro AP: {metrics['macro_average_precision']:.4f}")
    print(f"Model saved: {MODEL_DIR}")
    print("[OK] RoBERTa training completed")


if __name__ == "__main__":
    main()
