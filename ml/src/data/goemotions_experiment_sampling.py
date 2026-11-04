from collections import Counter

import torch

from ml.src.data.goemotions_sampling import (
    build_sampling_weights,
    sample_epoch_indices,
)

REAL_DRAWS = 43252
SYNTHETIC_QUOTAS = {
    "grief": 231,
    "nervousness": 492,
    "pride": 333,
    "relief": 459,
}


def combine_hybrid_schedules(
    rare_schedule,
    synthetic_schedule,
    seed,
    epoch,
):
    """
    Deterministic 50/50 mixture of two complete schedules.

    Each component contributes exactly half the epoch's
    draws. Sampling is without replacement from positions
    within each already-generated schedule.
    """
    if len(rare_schedule) != REAL_DRAWS or len(synthetic_schedule) != REAL_DRAWS:
        raise ValueError("Hybrid component length mismatch")

    if REAL_DRAWS % 2 != 0:
        raise ValueError("Hybrid requires even epoch length")

    half = REAL_DRAWS // 2

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch + 1_000_003)

    rare_positions = torch.randperm(
        REAL_DRAWS,
        generator=generator,
    )[:half].tolist()

    synthetic_positions = torch.randperm(
        REAL_DRAWS,
        generator=generator,
    )[:half].tolist()

    schedule = [rare_schedule[i] for i in rare_positions] + [
        synthetic_schedule[i] for i in synthetic_positions
    ]

    permutation = torch.randperm(
        REAL_DRAWS,
        generator=generator,
    ).tolist()

    return [schedule[i] for i in permutation]


def make_epoch_schedule(
    real_rows,
    synthetic_rows,
    label_names,
    strategy,
    seed,
    epoch,
):
    if len(real_rows) != REAL_DRAWS:
        raise ValueError("Unexpected real dataset size")

    if strategy not in (
        "uniform",
        "rare_oversample",
        "synthetic",
        "hybrid",
    ):
        raise ValueError("Unknown strategy")

    if seed < 0 or epoch < 0:
        raise ValueError("Invalid seed or epoch")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + epoch)

    if strategy in ("uniform", "rare_oversample"):
        weights = build_sampling_weights(
            real_rows,
            label_names,
            strategy,
        )
        indices = sample_epoch_indices(weights, seed, epoch)

        return [("real", int(index)) for index in indices.tolist()]

    synthetic_total = sum(SYNTHETIC_QUOTAS.values())
    real_count = REAL_DRAWS - synthetic_total

    if not synthetic_rows:
        raise ValueError("Synthetic rows required")

    groups = {label: [] for label in SYNTHETIC_QUOTAS}

    for index, row in enumerate(synthetic_rows):
        label = row["intended_primary_label"]

        if label not in groups:
            raise ValueError("Unexpected synthetic primary label")

        groups[label].append(index)

    if any(not group for group in groups.values()):
        raise ValueError("Missing synthetic target label")

    real_indices = torch.randint(
        len(real_rows),
        (real_count,),
        generator=generator,
    ).tolist()

    schedule = [("real", index) for index in real_indices]

    for label, quota in SYNTHETIC_QUOTAS.items():
        group = groups[label]

        positions = torch.randint(
            len(group),
            (quota,),
            generator=generator,
        ).tolist()

        schedule.extend(("synthetic", group[position]) for position in positions)

    if strategy == "hybrid":
        rare_schedule = make_epoch_schedule(
            real_rows=real_rows,
            synthetic_rows=synthetic_rows,
            label_names=label_names,
            strategy="rare_oversample",
            seed=seed,
            epoch=epoch,
        )

        synthetic_schedule = make_epoch_schedule(
            real_rows=real_rows,
            synthetic_rows=synthetic_rows,
            label_names=label_names,
            strategy="synthetic",
            seed=seed,
            epoch=epoch,
        )

        return combine_hybrid_schedules(
            rare_schedule,
            synthetic_schedule,
            seed,
            epoch,
        )

    permutation = torch.randperm(
        len(schedule),
        generator=generator,
    ).tolist()

    schedule = [schedule[i] for i in permutation]

    if len(schedule) != REAL_DRAWS:
        raise ValueError("Incorrect epoch length")

    return schedule


def summarize_epoch(
    schedule,
    real_rows,
    synthetic_rows,
    label_names,
):
    label_counts = Counter()
    source_counts = Counter()
    primary_counts = Counter()
    unique_ids = set()

    for source, index in schedule:
        rows = real_rows if source == "real" else synthetic_rows

        row = rows[index]
        source_counts[source] += 1
        label_counts.update(row["labels"])

        unique_ids.add((source, row["example_id"]))

        if source == "synthetic":
            primary_counts[row["intended_primary_label"]] += 1

    return {
        "total_draws": len(schedule),
        "source_draws": dict(source_counts),
        "unique_examples": len(unique_ids),
        "synthetic_primary_draws": dict(primary_counts),
        "label_occurrences": {
            label: label_counts[i] for i, label in enumerate(label_names)
        },
    }
