from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_QUEUE = Path("ml/artifacts/errors/textcnn_dair_v001_review_queue.csv")

DEFAULT_OUTPUT = Path("ml/artifacts/errors/textcnn_dair_v001_review_batch_001.csv")


BATCH_QUOTAS = {
    "all_models_same_wrong_prediction": 20,
    "high_score_textcnn_disagreement": 10,
    "baseline_correct_textcnn_wrong": 10,
    "other_textcnn_disagreement": 10,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the first controlled manual-review batch for TmotionLab Error Lab."
        )
    )

    parser.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def load_queue(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Review queue not found: {path}")

    frame = pd.read_csv(
        path,
        keep_default_na=False,
    )

    required = {
        "review_queue_rank",
        "example_id",
        "text",
        "true_label_name",
        "textcnn_prediction",
        "textcnn_score",
        "review_priority_group",
        "baseline_prediction",
        "numpy_prediction",
        "all_agree_and_wrong",
        "baseline_correct_textcnn_wrong",
        "three_way_disagreement",
        "review_outcome",
        "mechanism_tag",
        "review_notes",
        "reviewed_by",
        "review_status",
    }

    missing = required - set(frame.columns)

    if missing:
        raise ValueError(f"Review queue missing columns: {sorted(missing)}")

    if frame["example_id"].duplicated().any():
        raise ValueError("Review queue contains duplicate example_id")

    return frame


def select_group(
    frame: pd.DataFrame,
    *,
    group: str,
    count: int,
) -> pd.DataFrame:
    candidates = frame[frame["review_priority_group"] == group].copy()

    if len(candidates) < count:
        raise ValueError(
            f"Not enough examples for group {group}: "
            f"needed {count}, found {len(candidates)}"
        )

    # Queue is already prioritized.
    # Keep selection deterministic.
    return candidates.head(count)


def build_review_batch(
    queue: pd.DataFrame,
) -> pd.DataFrame:
    pieces = []

    for group, count in BATCH_QUOTAS.items():
        pieces.append(
            select_group(
                queue,
                group=group,
                count=count,
            )
        )

    batch = pd.concat(
        pieces,
        ignore_index=True,
    )

    if len(batch) != 50:
        raise ValueError(
            f"Review batch must contain exactly 50 examples, found {len(batch)}"
        )

    if batch["example_id"].duplicated().any():
        raise ValueError("Review batch contains duplicate examples")

    batch.insert(
        0,
        "batch_review_rank",
        range(
            1,
            len(batch) + 1,
        ),
    )

    return batch


def main() -> None:
    args = parse_args()

    queue = load_queue(args.queue)

    batch = build_review_batch(queue)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    batch.to_csv(
        args.output,
        index=False,
        encoding="utf-8",
    )

    print(f"[OK] Review batch created: {len(batch)} examples")

    print("[RESULT] Batch composition:")

    counts = batch["review_priority_group"].value_counts()

    for group, expected in BATCH_QUOTAS.items():
        actual = int(
            counts.get(
                group,
                0,
            )
        )

        print(f"  {group}: {actual} (expected {expected})")

    print(f"[OK] Review batch: {args.output}")


if __name__ == "__main__":
    main()
