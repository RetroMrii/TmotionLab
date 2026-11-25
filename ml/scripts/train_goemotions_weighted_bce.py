import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
from ml.src.data.goemotions_sampling import (
    build_sampling_weights,
)
from ml.scripts.train_goemotions_roberta import (
    compute_metrics,
    evaluate,
    make_targets,
    sigmoid,
)

ROOT = Path(__file__).resolve().parents[2]

CONFIG = ROOT / ("ml/configs/roberta_goemotions_multilabel_real_v001.json")

OUTPUT_ROOT = ROOT / "ml/artifacts/optimization/opt007"

SEED = 42
BASE_THRESHOLD = 0.25
ALPHA = 0.15
WEIGHT_CAP = 2.0


def calculate_positive_weights(rows, label_names):
    sampling = build_sampling_weights(rows, label_names, "rare_oversample")

    sample_weights = sampling.numpy()
    targets = make_targets(rows, len(label_names)).astype(np.float64)

    # Expected positive prevalence under weighted
    # sampling with replacement.
    prevalence = (sample_weights @ targets) / sample_weights.sum()

    prevalence = np.clip(prevalence, 1e-6, 1 - 1e-6)

    result = 1 + ALPHA * np.log((1 - prevalence) / prevalence)

    result = np.clip(result, 1.0, WEIGHT_CAP)

    return result.astype(np.float32), prevalence


class WeightedBCETrainer(Trainer):
    def __init__(self, *args, positive_weights, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_accepts_loss_kwargs = False

        self.positive_weights = torch.as_tensor(positive_weights, dtype=torch.float32)

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs["labels"].float()

        model_inputs = {key: value for key, value in inputs.items() if key != "labels"}

        outputs = model(**model_inputs)
        logits = outputs.logits.float()

        weights = self.positive_weights.to(
            device=logits.device,
            dtype=logits.dtype,
        )

        loss = F.binary_cross_entropy_with_logits(
            logits,
            labels,
            pos_weight=weights,
        )

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        required=True,
        choices=("baseline", "weighted"),
    )
    args = parser.parse_args()

    mode = args.mode
    run_dir = OUTPUT_ROOT / mode

    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {run_dir}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    train_config = config["training"]

    assert config["model_id"] == "FacebookAI/roberta-base"
    assert config["num_labels"] == 28
    assert config["max_length"] == 48
    assert train_config["num_train_epochs"] == 4

    set_seed(SEED)

    training_rows, label_names = load_goemotions_rows("train")
    validation_rows, validation_names = load_goemotions_rows("validation")

    assert len(training_rows) == 43252
    assert len(validation_rows) == 5426
    assert label_names == validation_names
    assert len(label_names) == 28

    calculated_weights, prevalence = calculate_positive_weights(
        training_rows, label_names
    )

    positive_weights = (
        np.ones(28, dtype=np.float32) if mode == "baseline" else calculated_weights
    )

    # Verify that unweighted BCE is recovered exactly
    # when every positive weight equals one.
    test_logits = torch.tensor([[0.2, -1.4], [0.8, 0.1]])
    test_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    regular_loss = F.binary_cross_entropy_with_logits(test_logits, test_targets)
    weighted_loss = F.binary_cross_entropy_with_logits(
        test_logits,
        test_targets,
        pos_weight=torch.ones(2),
    )

    assert torch.allclose(regular_loss, weighted_loss, atol=1e-7)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        use_fast=True,
    )

    train_dataset = GoEmotionsScheduledDataset(
        real_rows=training_rows,
        synthetic_rows=[],
        label_names=label_names,
        tokenizer=tokenizer,
        strategy="rare_oversample",
        seed=SEED,
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
        id2label={i: name for i, name in enumerate(label_names)},
        label2id={name: i for i, name in enumerate(label_names)},
        problem_type="multi_label_classification",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_id"],
        revision=config["model_revision"],
        config=model_config,
    )

    total_steps = (
        math.ceil(
            math.ceil(len(training_rows) / train_config["per_device_train_batch_size"])
            / train_config["gradient_accumulation_steps"]
        )
        * train_config["num_train_epochs"]
    )

    assert total_steps == 5408

    training_args = TrainingArguments(
        output_dir=str(run_dir / "checkpoints"),
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        learning_rate=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
        warmup_steps=train_config["warmup_ratio"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        fp16=train_config["fp16"],
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

    assert training_args.get_warmup_steps(total_steps) == 325

    trainer = WeightedBCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=GoEmotionsCollator(tokenizer),
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        positive_weights=positive_weights,
    )

    print("\nOPT-007")
    print("Mode:", mode)
    print("GPU:", torch.cuda.get_device_name(0))
    print("Optimizer steps:", total_steps)
    print("Warmup steps:", 325)
    print("\nPositive weights:")

    for name, weight, prior in zip(label_names, positive_weights, prevalence):
        print(f"{name:18} weight={weight:.3f} sampled_prevalence={prior:.5f}")

    print("\nStarting controlled training")

    train_result = trainer.train()

    assert trainer.state.global_step == total_steps
    assert trainer.state.best_model_checkpoint

    assert train_dataset.epoch_iteration_history == [0, 1, 2, 3]

    output = trainer.predict(validation_dataset)
    logits = output.predictions

    if isinstance(logits, tuple):
        logits = logits[0]

    probabilities = sigmoid(logits)
    truth = make_targets(validation_rows, 28)

    metrics = evaluate(
        truth,
        probabilities,
        BASE_THRESHOLD,
        label_names,
    )

    metrics["experiment"] = "OPT-007"
    metrics["mode"] = mode
    metrics["optimizer_steps"] = trainer.state.global_step
    metrics["best_checkpoint_macro_ap"] = trainer.state.best_metric
    metrics["positive_weights"] = {
        name: float(weight) for name, weight in zip(label_names, positive_weights)
    }

    trainer.save_model(str(run_dir / "best_model"))
    tokenizer.save_pretrained(str(run_dir / "best_model"))

    with (run_dir / "validation_predictions.jsonl").open(
        "x", encoding="utf-8"
    ) as stream:
        for i, row in enumerate(validation_rows):
            record = {
                "example_id": row["example_id"],
                "true_labels": row["labels"],
                "probabilities": probabilities[i].tolist(),
            }
            stream.write(json.dumps(record) + "\n")

    (run_dir / "validation_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    (run_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "seed": SEED,
                "strategy": "rare_oversample",
                "loss": "BCEWithLogitsLoss",
                "positive_weights": positive_weights.tolist(),
                "sampled_prevalence": prevalence.tolist(),
                "effective_warmup_steps": 325,
                "optimizer_steps": trainer.state.global_step,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_macro_ap": trainer.state.best_metric,
                "train_metrics": train_result.metrics,
                "training_history": trainer.state.log_history,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\nOPT-007 RESULTS")
    print("Mode:", mode)
    print("Macro F1:", metrics["macro_f1"])
    print("Micro F1:", metrics["micro_f1"])
    print("Macro AP:", metrics["macro_average_precision"])
    print("Saved:", run_dir)


if __name__ == "__main__":
    main()
