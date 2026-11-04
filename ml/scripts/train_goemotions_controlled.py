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

from ml.src.data.goemotions_dataset import (
    GoEmotionsCollator,
    GoEmotionsDataset,
    load_goemotions_rows,
)

from ml.src.data.goemotions_scheduled_dataset import (
    GoEmotionsScheduledDataset,
)

from ml.scripts.audit_goemotions_experiment_sampling import (
    load_synthetic,
)

from ml.scripts.train_goemotions_roberta import (
    compute_metrics,
    make_targets,
    sigmoid,
    choose_threshold,
    evaluate,
)
from ml.src.data.goemotions_experiment_sampling import (
    make_epoch_schedule,
)

from ml.scripts.audit_goemotions_multiseed_sampling import (
    schedule_digest,
)

ROOT = Path(__file__).resolve().parents[2]

BASE_CONFIG_PATH = ROOT / "ml/configs/roberta_goemotions_multilabel_real_v001.json"

DATA_MANIFEST_PATH = ROOT / "data/processed/goemotions_v001/manifest.json"

SYNTHETIC_MANIFEST_PATH = (
    ROOT / "data/processed/synthetic_goemotions_v002/manifest.json"
)

SAMPLING_AUDIT_PATH = ROOT / "reports/goemotions/controlled_sampling_audit_v001.json"

HYBRID_AUDIT_PATH = ROOT / "reports/goemotions/hybrid_sampling_audit_v001.json"

MULTISEED_AUDIT_PATH = ROOT / "reports/goemotions/multiseed_sampling_audit_v001.json"

EXPERIMENTS = {
    "uniform": "roberta_goemotions_uniform_v001",
    "rare_oversample": "roberta_goemotions_rare4x_v001",
    "synthetic": "roberta_goemotions_synthetic_v002",
    "hybrid": "roberta_goemotions_hybrid_v001",
}


def experiment_id_for(strategy, seed):
    if strategy not in EXPERIMENTS:
        raise ValueError(f"Unknown strategy: {strategy}")

    if seed not in (42, 123, 456):
        raise ValueError(f"Unapproved experimental seed: {seed}")

    original_id = EXPERIMENTS[strategy]

    if seed == 42:
        return original_id

    return f"{original_id}_seed{seed}"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


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


def expected_steps(num_examples, config):
    training = config["training"]

    batches = math.ceil(num_examples / training["per_device_train_batch_size"])

    updates_per_epoch = math.ceil(batches / training["gradient_accumulation_steps"])

    return updates_per_epoch * training["num_train_epochs"]


def build_paths(experiment):
    run_dir = ROOT / "ml/artifacts/models" / experiment

    return {
        "run_dir": run_dir,
        "checkpoint_dir": run_dir / "checkpoints",
        "model_dir": run_dir / "best_model",
        "metrics": (ROOT / "ml/artifacts/metrics" / f"{experiment}_validation.json"),
        "predictions": (
            ROOT / "ml/artifacts/predictions" / f"{experiment}_validation.jsonl"
        ),
        "log": (ROOT / "ml/artifacts/logs" / f"{experiment}_training.json"),
        "provenance": (run_dir / "experiment_provenance.json"),
    }


def verify_inputs(strategy, config, audit):
    if audit["experiment"] != "controlled_sampling_v001":
        raise ValueError("Wrong sampling audit")

    if (
        audit["seed"] != config["seed"]
        or audit["epochs"] != config["training"]["num_train_epochs"]
    ):
        raise ValueError("Sampling configuration mismatch")

    if strategy not in audit["strategies"]:
        raise ValueError("Missing audited strategy")

    if sha256_file(SYNTHETIC_MANIFEST_PATH) != audit["synthetic_manifest_sha256"]:
        raise ValueError("Synthetic dataset changed")

    if config["evaluation_split"] != "validation":
        raise ValueError("Unexpected evaluation split")

    if config["num_labels"] != 28:
        raise ValueError("Unexpected label count")


def verify_epoch_history(dataset, num_epochs):
    consumed = dataset.epoch_iteration_history
    expected = list(range(int(num_epochs)))

    if consumed != expected:
        raise RuntimeError(
            f"Unexpected epoch iteration sequence: {consumed}, expected {expected}"
        )

    print("[OK] Epochs consumed:", consumed)


def verify_multiseed_inputs(
    strategy,
    seed,
    config,
    audit,
    real_rows,
    synthetic_rows,
    label_names,
):
    if seed not in (123, 456):
        raise ValueError("Invalid multi-seed run")

    if audit["audit_id"] != "multiseed_sampling_v001":
        raise ValueError("Unexpected audit version")

    if seed not in audit["seeds"]:
        raise ValueError("Seed missing from audit")

    if strategy not in audit["strategies"]:
        raise ValueError("Strategy missing from audit")

    training = config["training"]
    epochs = int(training["num_train_epochs"])

    if (
        audit["epochs"] != epochs
        or audit["draws_per_epoch"] != len(real_rows)
        or len(real_rows) != 43252
    ):
        raise ValueError("Training budget mismatch")

    source_hashes = audit["source_hashes"]

    source_paths = {
        "synthetic_manifest": SYNTHETIC_MANIFEST_PATH,
        "controlled_sampling_audit": SAMPLING_AUDIT_PATH,
        "hybrid_sampling_audit": HYBRID_AUDIT_PATH,
    }

    for name, path in source_paths.items():
        if sha256_file(path) != source_hashes[name]:
            raise ValueError(f"Audit source changed: {name}")

    if (
        config["model_id"] != "FacebookAI/roberta-base"
        or config["model_revision"] != "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
        or config["num_labels"] != 28
        or config["evaluation_split"] != "validation"
    ):
        raise ValueError("Model specification mismatch")

    key = f"{strategy}_seed{seed}"
    audited_epochs = audit["runs"].get(key)

    if audited_epochs is None:
        raise ValueError(f"Missing audited run: {key}")

    if len(audited_epochs) != epochs:
        raise ValueError("Incorrect audited epoch count")

    for epoch, expected in enumerate(audited_epochs):
        schedule = make_epoch_schedule(
            real_rows,
            synthetic_rows,
            label_names,
            strategy,
            seed,
            epoch,
        )

        actual_digest = schedule_digest(schedule)

        if actual_digest != expected["schedule_sha256"]:
            raise ValueError(f"Schedule mismatch: {key}, epoch {epoch + 1}")

        if len(schedule) != 43252:
            raise ValueError("Unexpected epoch length")

        print(f"[AUDIT] {key} epoch={epoch + 1} SHA256={actual_digest[:16]}... OK")

    print(f"[OK] All {epochs} epoch schedules verified for {key}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--strategy",
        required=True,
        choices=tuple(EXPERIMENTS),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        choices=(42, 123, 456),
    )

    args = parser.parse_args()

    strategy = args.strategy
    seed = args.seed
    experiment = experiment_id_for(strategy, seed)

    config = load_json(BASE_CONFIG_PATH)
    manifest = load_json(DATA_MANIFEST_PATH)

    if seed == 42:
        audit_path = HYBRID_AUDIT_PATH if strategy == "hybrid" else SAMPLING_AUDIT_PATH
        audit = load_json(audit_path)

        # Keep your existing seed-42 verification logic here.

    else:
        audit_path = MULTISEED_AUDIT_PATH
        audit = load_json(audit_path)

    if seed == 42:
        audit_path = HYBRID_AUDIT_PATH if strategy == "hybrid" else SAMPLING_AUDIT_PATH

        audit = load_json(audit_path)

        if strategy == "hybrid":
            if (
                audit["experiment"] != "hybrid_sampling_v001"
                or audit["strategy"] != "hybrid"
                or audit["seed"] != seed
                or audit["epochs"] != config["training"]["num_train_epochs"]
                or audit["draws_per_epoch"] != 43252
                or len(audit["epoch_reports"]) != config["training"]["num_train_epochs"]
                or sha256_file(SAMPLING_AUDIT_PATH) != audit["previous_audit_sha256"]
                or sha256_file(SYNTHETIC_MANIFEST_PATH)
                != audit["synthetic_manifest_sha256"]
            ):
                raise ValueError("Hybrid audit mismatch")

            verify_inputs(
                "synthetic",
                config,
                load_json(SAMPLING_AUDIT_PATH),
            )
        else:
            verify_inputs(strategy, config, audit)

    else:
        audit_path = MULTISEED_AUDIT_PATH
        audit = load_json(audit_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    paths = build_paths(experiment)

    for key in ("run_dir", "metrics", "predictions", "log"):
        if paths[key].exists():
            raise FileExistsError(f"Experiment artifact exists: {paths[key]}")

    set_seed(seed)

    real_rows, label_names = load_goemotions_rows("train")
    validation_rows, validation_names = load_goemotions_rows("validation")

    if label_names != validation_names:
        raise ValueError("Label mapping mismatch")

    synthetic_rows, _ = load_synthetic()

    if seed != 42:
        verify_multiseed_inputs(
            strategy=strategy,
            seed=seed,
            config=config,
            audit=audit,
            real_rows=real_rows,
            synthetic_rows=synthetic_rows,
            label_names=label_names,
        )

    if len(real_rows) != 43252:
        raise ValueError("Unexpected real training size")

    if len(validation_rows) != 5426:
        raise ValueError("Unexpected validation size")

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        use_fast=True,
    )

    train_dataset = GoEmotionsScheduledDataset(
        real_rows=real_rows,
        synthetic_rows=synthetic_rows,
        label_names=label_names,
        tokenizer=tokenizer,
        strategy=strategy,
        seed=seed,
        max_length=config["max_length"],
    )

    validation_dataset = GoEmotionsDataset(
        validation_rows,
        tokenizer,
        max_length=config["max_length"],
    )

    model_config = AutoConfig.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        num_labels=28,
        id2label={i: label for i, label in enumerate(label_names)},
        label2id={label: i for i, label in enumerate(label_names)},
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        config=model_config,
    )

    training = config["training"]

    paths["run_dir"].mkdir(
        parents=True,
        exist_ok=False,
    )

    training_args = TrainingArguments(
        output_dir=str(paths["checkpoint_dir"]),
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=(training["gradient_accumulation_steps"]),
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        warmup_steps=training["warmup_ratio"],
        lr_scheduler_type=training["lr_scheduler_type"],
        fp16=training["fp16"],
        seed=seed,
        data_seed=seed,
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
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=GoEmotionsCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    provenance = {
        "experiment_id": experiment,
        "strategy": strategy,
        "base_config_sha256": sha256_file(BASE_CONFIG_PATH),
        "real_dataset_manifest_sha256": sha256_file(DATA_MANIFEST_PATH),
        "synthetic_dataset_manifest_sha256": sha256_file(SYNTHETIC_MANIFEST_PATH),
        "sampling_policy": strategy,
        "hybrid_components": (
            {
                "rare_oversample_fraction": 0.5,
                "synthetic_policy_fraction": 0.5,
            }
            if strategy == "hybrid"
            else None
        ),
        "model_id": config["model_id"],
        "model_revision": config["model_revision"],
        "seed": seed,
        "sampling_audit_sha256": sha256_file(audit_path),
        "audited_run_key": (f"{strategy}_seed{seed}" if seed != 42 else None),
        "expected_optimizer_steps": expected_steps(len(real_rows), config),
        "test_evaluated": False,
    }

    save_json(paths["provenance"], provenance)

    print("Experiment:", experiment)
    print("Sampling strategy:", strategy)
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "Expected optimizer steps:",
        provenance["expected_optimizer_steps"],
    )

    result = trainer.train()

    verify_epoch_history(
        train_dataset,
        training["num_train_epochs"],
    )

    if trainer.state.global_step != (provenance["expected_optimizer_steps"]):
        raise RuntimeError(
            "Optimizer-step mismatch: "
            f"actual={trainer.state.global_step}, "
            f"expected="
            f"{provenance['expected_optimizer_steps']}"
        )

    if not trainer.state.best_model_checkpoint:
        raise RuntimeError("Best checkpoint missing")

    output = trainer.predict(validation_dataset)

    logits = output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = sigmoid(logits)
    truth = make_targets(validation_rows, 28)

    if probabilities.shape != truth.shape:
        raise ValueError("Prediction shape mismatch")

    threshold, threshold_search = choose_threshold(
        truth,
        probabilities,
        config["threshold_candidates"],
    )

    metrics = evaluate(
        truth,
        probabilities,
        threshold,
        label_names,
    )

    predicted = (probabilities >= threshold).astype(np.int8)

    metrics.update(
        {
            "experiment_id": experiment,
            "strategy": strategy,
            "evaluation_split": "validation",
            "test_evaluated": False,
            "selected_threshold": threshold,
            "threshold_search": threshold_search,
            "best_checkpoint": (Path(trainer.state.best_model_checkpoint).name),
            "best_checkpoint_macro_ap": (trainer.state.best_metric),
            "optimizer_steps": trainer.state.global_step,
            "epoch_iteration_history": (train_dataset.epoch_iteration_history),
            "sampling_audit_sha256": provenance["sampling_audit_sha256"],
        }
    )

    trainer.save_model(str(paths["model_dir"]))
    tokenizer.save_pretrained(str(paths["model_dir"]))

    paths["predictions"].parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with paths["predictions"].open("x", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(validation_rows):
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
        path.name: sha256_file(path)
        for path in paths["model_dir"].iterdir()
        if path.is_file()
    }

    save_json(paths["metrics"], metrics)

    save_json(
        paths["log"],
        {
            "experiment_id": experiment,
            "strategy": strategy,
            "training_metrics": result.metrics,
            "best_metric": trainer.state.best_metric,
            "best_checkpoint": (trainer.state.best_model_checkpoint),
            "history": trainer.state.log_history,
            "epoch_set_history": (train_dataset.epoch_set_history),
            "epoch_iteration_history": (train_dataset.epoch_iteration_history),
            "optimizer_steps": trainer.state.global_step,
        },
    )

    print("\nTRAINING RESULTS")
    print("Experiment:", experiment)
    print("Threshold:", threshold)
    print("Macro F1:", metrics["macro_f1"])
    print("Micro F1:", metrics["micro_f1"])
    print(
        "Macro AP:",
        metrics["macro_average_precision"],
    )
    print(
        "Optimizer steps:",
        trainer.state.global_step,
    )
    print("[OK] Controlled training completed")


if __name__ == "__main__":
    main()
