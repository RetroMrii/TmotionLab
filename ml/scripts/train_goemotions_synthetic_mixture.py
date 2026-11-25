import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

from ml.src.data.goemotions_dataset import (
    GoEmotionsCollator,
    GoEmotionsDataset,
    load_goemotions_rows,
)
from ml.scripts.train_goemotions_roberta import (
    compute_metrics,
    evaluate,
    make_targets,
    sigmoid,
)

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / ("ml/configs/roberta_goemotions_multilabel_real_v001.json")
REAL_MANIFEST = ROOT / "data/processed/goemotions_v001/manifest.json"

SYNTHETIC_DIR = ROOT / ("data/processed/synthetic_goemotions_opt009_v001")
SYNTHETIC_DATA = SYNTHETIC_DIR / "train.jsonl"
SYNTHETIC_MANIFEST = SYNTHETIC_DIR / "manifest.json"

OUTPUT_ROOT = ROOT / "ml/artifacts/optimization/opt010"

FRACTIONS = {
    "control": 0.00,
    "synthetic10": 0.10,
    "synthetic25": 0.25,
    "synthetic40": 0.40,
}

RARE_LABELS = {"grief", "nervousness", "pride", "relief"}
RARE_MULTIPLIER = 4.0
SEED = 42
THRESHOLD = 0.25
EXPECTED_TRAIN = 43252
EXPECTED_VALIDATION = 5426
EXPECTED_SYNTHETIC = 41927
EXPECTED_STEPS = 5408


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
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


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_inputs():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    real_manifest = json.loads(REAL_MANIFEST.read_text(encoding="utf-8"))
    synthetic_manifest = json.loads(SYNTHETIC_MANIFEST.read_text(encoding="utf-8"))

    if synthetic_manifest["training_data_sha256"] != sha256_file(SYNTHETIC_DATA):
        raise ValueError("OPT-009 synthetic data hash mismatch")

    if synthetic_manifest["retained"] != EXPECTED_SYNTHETIC:
        raise ValueError("Unexpected synthetic manifest count")

    if synthetic_manifest["training_approval"] is not False:
        raise ValueError("Unexpected synthetic approval state")

    if config["model_id"] != "FacebookAI/roberta-base":
        raise ValueError("Unexpected model family")

    if config["num_labels"] != 28:
        raise ValueError("Expected 28 labels")

    if config["evaluation_split"] != "validation":
        raise ValueError("Evaluation split changed")

    real, labels = load_goemotions_rows("train")
    validation, validation_labels = load_goemotions_rows("validation")

    if labels != validation_labels:
        raise ValueError("Label vocabulary mismatch")

    if labels != real_manifest["label_names"]:
        raise ValueError("Real dataset manifest mismatch")

    if len(real) != EXPECTED_TRAIN:
        raise ValueError("Real training size changed")

    if len(validation) != EXPECTED_VALIDATION:
        raise ValueError("Validation size changed")

    synthetic = load_jsonl(SYNTHETIC_DATA)

    if len(synthetic) != EXPECTED_SYNTHETIC:
        raise ValueError("Unexpected synthetic row count")

    ids = set()

    for row in synthetic:
        identifier = row["example_id"]

        if identifier in ids:
            raise ValueError("Duplicate synthetic example ID")

        ids.add(identifier)

        row_labels = row["labels"]

        if (
            not isinstance(row_labels, list)
            or not row_labels
            or len(row_labels) != len(set(row_labels))
            or any(
                type(value) is not int or value < 0 or value >= 28
                for value in row_labels
            )
        ):
            raise ValueError(f"Invalid synthetic labels: {identifier}")

        if row.get("annotation_verified") is not False:
            raise ValueError("Unexpected annotation provenance")

    return config, real, validation, synthetic, labels


class MixedScheduledDataset(IterableDataset):
    """
    Fixed-length, deterministic epoch schedules.

    Real examples follow the rare4x row-weighting policy.
    Synthetic examples are sampled uniformly with replacement.

    Each epoch has exactly len(real_rows) draws, irrespective
    of the synthetic fraction.
    """

    def __init__(
        self,
        real_rows,
        synthetic_rows,
        labels,
        tokenizer,
        max_length,
        synthetic_fraction,
        seed,
    ):
        super().__init__()

        self.real_rows = real_rows
        self.synthetic_rows = synthetic_rows
        self.real_dataset = GoEmotionsDataset(
            real_rows, tokenizer, max_length=max_length
        )
        self.synthetic_dataset = GoEmotionsDataset(
            synthetic_rows, tokenizer, max_length=max_length
        )

        self.length = len(real_rows)
        self.synthetic_fraction = synthetic_fraction
        self.seed = seed
        self.next_epoch = 0
        self.epoch_iteration_history = []
        self.rare_ids = {labels.index(name) for name in RARE_LABELS}

        weights = [
            RARE_MULTIPLIER
            if any(label in self.rare_ids for label in row["labels"])
            else 1.0
            for row in real_rows
        ]
        self.real_weights = torch.tensor(weights, dtype=torch.double)

    def __len__(self):
        return self.length

    def epoch_schedule(self, epoch):
        # round-half-up for nonnegative draw counts
        synthetic_count = int(math.floor(self.length * self.synthetic_fraction + 0.5))
        real_count = self.length - synthetic_count

        # Preserve the original rare4x real-sampler seed
        # across the control and synthetic-mixture runs.
        real_generator = torch.Generator().manual_seed(self.seed + epoch)

        real_indices = torch.multinomial(
            self.real_weights,
            num_samples=real_count,
            replacement=True,
            generator=real_generator,
        ).tolist()

        synthetic_generator = torch.Generator().manual_seed(self.seed + 100000 + epoch)

        synthetic_indices = torch.randint(
            high=len(self.synthetic_rows),
            size=(synthetic_count,),
            generator=synthetic_generator,
        ).tolist()

        schedule = [(0, index) for index in real_indices] + [
            (1, index) for index in synthetic_indices
        ]

        # Shuffle positions, independently from row selection.
        shuffle_generator = torch.Generator().manual_seed(self.seed + 200000 + epoch)
        permutation = torch.randperm(
            len(schedule), generator=shuffle_generator
        ).tolist()

        return [schedule[index] for index in permutation]

    def __iter__(self):
        epoch = self.next_epoch
        self.next_epoch += 1
        self.epoch_iteration_history.append(epoch)

        for source, index in self.epoch_schedule(epoch):
            if source == 0:
                yield self.real_dataset[index]
            else:
                yield self.synthetic_dataset[index]


def schedule_digest(schedule):
    payload = json.dumps(schedule, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_preflight(dataset, epochs):
    result = []

    for epoch in range(epochs):
        schedule = dataset.epoch_schedule(epoch)
        real_count = sum(source == 0 for source, _ in schedule)
        synthetic_count = len(schedule) - real_count

        assert len(schedule) == EXPECTED_TRAIN

        result.append(
            {
                "epoch": epoch,
                "real_draws": real_count,
                "synthetic_draws": synthetic_count,
                "schedule_sha256": schedule_digest(schedule),
            }
        )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=FRACTIONS, required=True)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()

    config, real, validation, synthetic, labels = load_inputs()

    training = config["training"]
    fraction = FRACTIONS[args.mode]
    epochs = int(training["num_train_epochs"])

    if epochs != 4:
        raise ValueError("Unexpected epoch budget")

    steps_per_epoch = math.ceil(
        math.ceil(EXPECTED_TRAIN / training["per_device_train_batch_size"])
        / training["gradient_accumulation_steps"]
    )
    total_steps = steps_per_epoch * epochs

    if total_steps != EXPECTED_STEPS:
        raise ValueError(f"Optimizer-step budget changed: {total_steps}")

    # Planning uses no model weights and no GPU training.
    if args.plan:
        synthetic_draws = int(math.floor(EXPECTED_TRAIN * fraction + 0.5))
        print("OPT-010 PLAN")
        print("Mode:", args.mode)
        print("Real rows:", len(real))
        print("Synthetic pool:", len(synthetic))
        print("Validation:", len(validation))
        print("Real draws per epoch:", EXPECTED_TRAIN - synthetic_draws)
        print("Synthetic draws per epoch:", synthetic_draws)
        print("Epochs:", epochs)
        print("Optimizer steps:", total_steps)
        print("No training performed.")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    run_dir = OUTPUT_ROOT / args.mode

    # Never overwrite completed or partially completed runs.
    if run_dir.exists():
        raise FileExistsError(f"OPT-010 output already exists: {run_dir}")

    set_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        use_fast=True,
    )

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

    schedules = run_preflight(train_dataset, epochs)

    model_config = AutoConfig.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        num_labels=28,
        id2label={index: label for index, label in enumerate(labels)},
        label2id={label: index for index, label in enumerate(labels)},
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        config=model_config,
    )

    # Create the run directory before initializing Trainer,
    # avoiding the output initialization issue from OPT-007.
    run_dir.mkdir(parents=True, exist_ok=False)

    save_json(
        run_dir / "provenance.json",
        {
            "experiment": "OPT-010",
            "mode": args.mode,
            "synthetic_fraction": fraction,
            "seed": SEED,
            "real_sampler": "rare4x",
            "synthetic_sampler": "uniform_replacement",
            "loss": "standard_BCE",
            "model_id": config["model_id"],
            "model_revision": config["model_revision"],
            "config_sha256": sha256_file(CONFIG_PATH),
            "real_manifest_sha256": sha256_file(REAL_MANIFEST),
            "synthetic_manifest_sha256": sha256_file(SYNTHETIC_MANIFEST),
            "synthetic_data_sha256": sha256_file(SYNTHETIC_DATA),
            "epochs": epochs,
            "expected_optimizer_steps": total_steps,
            "schedule_audit": schedules,
            "threshold": THRESHOLD,
            "evaluation_split": "validation",
            "test_evaluated": False,
        },
    )

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=(training["gradient_accumulation_steps"]),
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_steps=training["warmup_ratio"],
        lr_scheduler_type=training["lr_scheduler_type"],
        fp16=training["fp16"],
        seed=SEED,
        data_seed=SEED,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_average_precision",
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

    print("\nOPT-010 TRAINING")
    print("Mode:", args.mode)
    print("Synthetic fraction:", fraction)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Optimizer steps:", total_steps)
    print("Warmup steps:", training_args.get_warmup_steps(total_steps))

    result = trainer.train()

    if trainer.state.global_step != total_steps:
        raise RuntimeError(
            f"Optimizer-step mismatch: {trainer.state.global_step} != {total_steps}"
        )

    expected_history = list(range(epochs))
    if train_dataset.epoch_iteration_history != expected_history:
        raise RuntimeError(
            f"Unexpected epoch consumption: {train_dataset.epoch_iteration_history}"
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
        raise ValueError("Prediction shape mismatch")

    metrics = evaluate(truth, probabilities, THRESHOLD, labels)

    metrics.update(
        {
            "experiment": "OPT-010",
            "mode": args.mode,
            "synthetic_fraction": fraction,
            "evaluation_split": "validation",
            "threshold": THRESHOLD,
            "optimizer_steps": trainer.state.global_step,
            "best_checkpoint_macro_ap": trainer.state.best_metric,
            "best_checkpoint": Path(trainer.state.best_model_checkpoint).name,
            "test_evaluated": False,
        }
    )

    trainer.save_model(str(run_dir / "best_model"))
    tokenizer.save_pretrained(str(run_dir / "best_model"))

    predictions_path = run_dir / "validation_predictions.jsonl"

    predicted = probabilities >= THRESHOLD

    with predictions_path.open("x", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(validation):
            record = {
                "example_id": row["example_id"],
                "true_labels": row["labels"],
                "predicted_labels": np.flatnonzero(predicted[index]).tolist(),
                "probabilities": probabilities[index].tolist(),
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
            "training_metrics": result.metrics,
            "history": trainer.state.log_history,
            "epoch_iteration_history": train_dataset.epoch_iteration_history,
            "optimizer_steps": trainer.state.global_step,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_macro_ap": trainer.state.best_metric,
        },
    )

    print("\nOPT-010 RESULTS")
    print("Mode:", args.mode)
    print("Macro F1:", metrics["macro_f1"])
    print("Micro F1:", metrics["micro_f1"])
    print("Macro AP:", metrics["macro_average_precision"])
    print("Optimizer steps:", trainer.state.global_step)
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
