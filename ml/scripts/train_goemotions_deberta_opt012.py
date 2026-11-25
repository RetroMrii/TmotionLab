import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ml.scripts.train_goemotions_synthetic_mixture import (
    FRACTIONS,
    MixedScheduledDataset,
    load_inputs,
    run_preflight,
)
from ml.src.data.goemotions_dataset import (
    GoEmotionsCollator,
    GoEmotionsDataset,
)
from ml.scripts.train_goemotions_roberta import (
    compute_metrics,
    evaluate,
    make_targets,
    sigmoid,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "microsoft/deberta-v3-base"
OUTPUT_ROOT = ROOT / "ml/artifacts/optimization/opt012"
CONFIG_PATH = ROOT / ("ml/configs/roberta_goemotions_multilabel_real_v001.json")
SYNTHETIC_PATH = ROOT / ("data/processed/synthetic_goemotions_opt009_v001/train.jsonl")

SEED = 42
THRESHOLD = 0.25
EXPECTED_DRAWS = 43252
EXPECTED_STEPS = 5408


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            data,
            stream,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("control", "synthetic10"),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Reduce activation memory usage at the cost of speed.",
    )
    args = parser.parse_args()

    config, real, validation, synthetic, labels = load_inputs()

    training = config["training"]
    fraction = FRACTIONS[args.mode]
    epochs = int(training["num_train_epochs"])
    batch_size = int(training["per_device_train_batch_size"])
    accumulation = int(training["gradient_accumulation_steps"])

    steps_per_epoch = math.ceil(math.ceil(EXPECTED_DRAWS / batch_size) / accumulation)
    total_steps = steps_per_epoch * epochs

    if epochs != 4 or total_steps != EXPECTED_STEPS or len(real) != EXPECTED_DRAWS:
        raise ValueError("OPT-012 training budget mismatch")

    synthetic_count = math.floor(EXPECTED_DRAWS * fraction + 0.5)

    print("\nOPT-012 — DEBERTA")
    print("Mode:", args.mode)
    print("Model:", MODEL_ID)
    print("Real examples:", len(real))
    print("Synthetic pool:", len(synthetic))
    print("Validation:", len(validation))
    print("Real draws per epoch:", EXPECTED_DRAWS - synthetic_count)
    print("Synthetic draws per epoch:", synthetic_count)
    print("Epochs:", epochs)
    print("Optimizer steps:", total_steps)
    print("Gradient checkpointing:", args.gradient_checkpointing)

    if args.plan:
        print("Planning complete. No training.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    run_dir = OUTPUT_ROOT / args.mode

    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
    )

    model_config = AutoConfig.from_pretrained(
        MODEL_ID,
        num_labels=28,
        id2label={i: label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=model_config,
        dtype=torch.float32,
    )

    # Under AMP, keep trainable model parameters in FP32.
    # FP16 is used for eligible computation, not stored weights.
    model.float()

    parameter_dtypes = {
        str(param.dtype) for param in model.parameters() if param.requires_grad
    }

    print("Trainable parameter dtypes:", parameter_dtypes)

    if parameter_dtypes != {"torch.float32"}:
        raise RuntimeError(
            "OPT-012 requires FP32 trainable parameters before initializing Trainer"
        )

    # DeBERTa-v3-base is implemented through
    # the Transformers DeBERTa-v2 model classes.
    print("Model class:", type(model).__name__)
    print("Tokenizer class:", type(tokenizer).__name__)

    train_dataset = MixedScheduledDataset(
        real_rows=real,
        synthetic_rows=synthetic,
        labels=labels,
        tokenizer=tokenizer,
        max_length=config["max_length"],
        synthetic_fraction=fraction,
        seed=SEED,
    )

    validation_dataset = GoEmotionsDataset(
        validation,
        tokenizer,
        max_length=config["max_length"],
    )

    schedule_audit = run_preflight(train_dataset, epochs)

    run_dir.mkdir(parents=True, exist_ok=False)

    provenance = {
        "experiment": "OPT-012",
        "mode": args.mode,
        "architecture": "DeBERTa-v3-base",
        "model_id": MODEL_ID,
        "resolved_model_revision": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_revision": (tokenizer.init_kwargs.get("_commit_hash")),
        "model_class": type(model).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "synthetic_fraction": fraction,
        "seed": SEED,
        "real_sampler": "rare4x",
        "synthetic_sampler": "uniform_replacement",
        "loss": "standard_BCE",
        "threshold": THRESHOLD,
        "config_sha256": sha256_file(CONFIG_PATH),
        "synthetic_data_sha256": sha256_file(SYNTHETIC_PATH),
        "gradient_checkpointing": (args.gradient_checkpointing),
        "expected_optimizer_steps": total_steps,
        "schedule_audit": schedule_audit,
        "evaluation_split": "validation",
        "test_evaluated": False,
    }

    save_json(
        run_dir / "provenance.json",
        provenance,
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=accumulation,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_steps=training["warmup_ratio"],
        lr_scheduler_type=training["lr_scheduler_type"],
        fp16=training["fp16"],
        gradient_checkpointing=(args.gradient_checkpointing),
        seed=SEED,
        data_seed=SEED,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=("macro_average_precision"),
        greater_is_better=True,
        save_total_limit=1,
        eval_accumulation_steps=16,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=GoEmotionsCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Starting training.")

    result = trainer.train()

    if trainer.state.global_step != total_steps:
        raise RuntimeError(
            f"Optimizer steps mismatch: {trainer.state.global_step} != {total_steps}"
        )

    if train_dataset.epoch_iteration_history != (list(range(epochs))):
        raise RuntimeError(
            f"Epoch consumption mismatch: {train_dataset.epoch_iteration_history}"
        )

    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("Best checkpoint missing")

    prediction_output = trainer.predict(validation_dataset)

    logits = prediction_output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = sigmoid(logits)
    truth = make_targets(validation, 28)

    if probabilities.shape != truth.shape:
        raise ValueError("Validation prediction shape mismatch")

    metrics = evaluate(
        truth,
        probabilities,
        THRESHOLD,
        labels,
    )

    metrics.update(
        {
            "experiment": "OPT-012",
            "mode": args.mode,
            "model_id": MODEL_ID,
            "synthetic_fraction": fraction,
            "threshold": THRESHOLD,
            "evaluation_split": "validation",
            "optimizer_steps": (trainer.state.global_step),
            "best_checkpoint_macro_ap": (trainer.state.best_metric),
            "best_checkpoint": Path(trainer.state.best_model_checkpoint).name,
            "test_evaluated": False,
        }
    )

    trainer.save_model(str(run_dir / "best_model"))
    tokenizer.save_pretrained(str(run_dir / "best_model"))

    predictions = probabilities >= THRESHOLD

    with (run_dir / "validation_predictions.jsonl").open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        for index, row in enumerate(validation):
            record = {
                "example_id": row["example_id"],
                "true_labels": row["labels"],
                "predicted_labels": (np.flatnonzero(predictions[index]).tolist()),
                "probabilities": (probabilities[index].tolist()),
            }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    save_json(
        run_dir / "validation_metrics.json",
        metrics,
    )

    save_json(
        run_dir / "training_summary.json",
        {
            "mode": args.mode,
            "model_id": MODEL_ID,
            "training_metrics": result.metrics,
            "history": trainer.state.log_history,
            "epoch_iteration_history": (train_dataset.epoch_iteration_history),
            "optimizer_steps": (trainer.state.global_step),
            "best_checkpoint": (trainer.state.best_model_checkpoint),
            "best_macro_ap": (trainer.state.best_metric),
        },
    )

    print("\nOPT-012 RESULTS")
    print("Mode:", args.mode)
    print("Macro AP:", metrics["macro_average_precision"])
    print("Macro F1:", metrics["macro_f1"])
    print("Micro F1:", metrics["micro_f1"])
    print("Optimizer steps:", trainer.state.global_step)
    print("Saved:", run_dir)
    print("Production approval: NOT GRANTED")


if __name__ == "__main__":
    main()
