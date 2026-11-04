from collections import Counter

import torch


TARGET_LABELS = (
    "grief",
    "nervousness",
    "pride",
    "relief",
)

BOOST_FACTOR = 4.0


def build_sampling_weights(
    rows,
    label_names,
    strategy,
    boost_factor=BOOST_FACTOR,
):
    if strategy not in ("uniform", "rare_oversample"):
        raise ValueError(f"Unknown strategy: {strategy}")

    if not rows:
        raise ValueError("Training rows cannot be empty")

    if boost_factor < 1:
        raise ValueError("Boost factor must be >= 1")

    label_to_id = {name: index for index, name in enumerate(label_names)}

    missing = set(TARGET_LABELS) - set(label_to_id)
    if missing:
        raise ValueError(f"Missing labels: {sorted(missing)}")

    target_ids = {label_to_id[name] for name in TARGET_LABELS}

    weights = torch.ones(
        len(rows),
        dtype=torch.float64,
    )

    if strategy == "rare_oversample":
        for index, row in enumerate(rows):
            if target_ids.intersection(row["labels"]):
                weights[index] = boost_factor

    return weights


def sample_epoch_indices(
    weights,
    seed,
    epoch,
):
    if weights.ndim != 1 or not len(weights):
        raise ValueError("Invalid weight vector")

    if not torch.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Weights must be finite and positive")

    if seed < 0 or epoch < 0:
        raise ValueError("Seed and epoch must be nonnegative")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)

    return torch.multinomial(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=generator,
    )


def summarize_sampling(
    rows,
    label_names,
    weights,
    sampled_indices,
):
    if len(weights) != len(rows):
        raise ValueError("Weight count mismatch")

    if len(sampled_indices) != len(rows):
        raise ValueError("Sample count mismatch")

    label_to_id = {name: index for index, name in enumerate(label_names)}

    source_counts = Counter()
    sampled_counts = Counter()

    for row in rows:
        source_counts.update(row["labels"])

    for index in sampled_indices.tolist():
        sampled_counts.update(rows[index]["labels"])

    target_report = {}

    for name in TARGET_LABELS:
        label_id = label_to_id[name]

        original = source_counts[label_id]
        sampled = sampled_counts[label_id]

        target_report[name] = {
            "original_positive_occurrences": original,
            "sampled_positive_occurrences": sampled,
            "exposure_ratio": (sampled / original if original else None),
        }

    return {
        "training_examples": len(rows),
        "epoch_draws": len(sampled_indices),
        "unique_examples_sampled": int(torch.unique(sampled_indices).numel()),
        "target_labels": target_report,
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
    }
