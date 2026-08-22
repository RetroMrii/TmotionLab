from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze behavioral differences between TmotionLab models."
    )

    parser.add_argument(
        "--comparison",
        type=Path,
        default=Path("ml/artifacts/comparisons/dair_v001_model_comparison.csv"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml/artifacts/comparisons"),
    )

    return parser.parse_args()


def confusion_counts(
    frame: pd.DataFrame,
    *,
    prediction_column: str,
) -> list[dict]:
    if frame.empty:
        return []

    grouped = (
        frame.groupby(
            [
                "true_label_name",
                prediction_column,
            ]
        )
        .size()
        .reset_index(name="count")
        .sort_values(
            "count",
            ascending=False,
        )
    )

    return grouped.to_dict(orient="records")


def class_counts(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "true_label_name",
                "count",
            ]
        )

    return (
        frame["true_label_name"]
        .value_counts()
        .rename_axis("true_label_name")
        .reset_index(name="count")
    )


def rate(
    numerator: int,
    denominator: int,
) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def main() -> None:
    args = parse_args()

    if not args.comparison.exists():
        raise FileNotFoundError(f"Comparison artifact not found: {args.comparison}")

    frame = pd.read_csv(args.comparison)

    required_columns = {
        "example_id",
        "text",
        "true_label_name",
        "baseline_prediction",
        "numpy_prediction",
        "textcnn_prediction",
        "baseline_correct",
        "numpy_correct",
        "textcnn_correct",
        "baseline_wrong_textcnn_correct",
        "baseline_correct_textcnn_wrong",
        "all_agree_and_wrong",
        "three_way_disagreement",
    }

    missing = required_columns - set(frame.columns)

    if missing:
        raise ValueError(f"Missing comparison columns: {sorted(missing)}")

    output_dir = args.output_dir
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    textcnn_fixes = frame[frame["baseline_wrong_textcnn_correct"]].copy()

    textcnn_regressions = frame[frame["baseline_correct_textcnn_wrong"]].copy()

    shared_errors = frame[frame["all_agree_and_wrong"]].copy()

    disagreements = frame[frame["three_way_disagreement"]].copy()

    fixes_by_class = class_counts(textcnn_fixes)

    regressions_by_class = class_counts(textcnn_regressions)

    fixes_by_class.to_csv(
        output_dir / "dair_v001_textcnn_fixes_by_class.csv",
        index=False,
    )

    regressions_by_class.to_csv(
        output_dir / "dair_v001_textcnn_regressions_by_class.csv",
        index=False,
    )

    shared_errors.to_csv(
        output_dir / "dair_v001_shared_errors.csv",
        index=False,
    )

    disagreements.to_csv(
        output_dir / "dair_v001_three_way_disagreements.csv",
        index=False,
    )

    total = len(frame)

    summary = {
        "num_examples": total,
        "cohorts": {
            "textcnn_fixes_baseline": {
                "count": int(len(textcnn_fixes)),
                "rate": rate(
                    len(textcnn_fixes),
                    total,
                ),
                "by_true_class": (fixes_by_class.to_dict(orient="records")),
                "baseline_confusions_fixed": (
                    confusion_counts(
                        textcnn_fixes,
                        prediction_column=("baseline_prediction"),
                    )
                ),
            },
            "baseline_beats_textcnn": {
                "count": int(len(textcnn_regressions)),
                "rate": rate(
                    len(textcnn_regressions),
                    total,
                ),
                "by_true_class": (regressions_by_class.to_dict(orient="records")),
                "textcnn_regressions": (
                    confusion_counts(
                        textcnn_regressions,
                        prediction_column=("textcnn_prediction"),
                    )
                ),
            },
            "all_agree_and_wrong": {
                "count": int(len(shared_errors)),
                "rate": rate(
                    len(shared_errors),
                    total,
                ),
                "by_true_class": (
                    class_counts(shared_errors).to_dict(orient="records")
                ),
                "shared_confusions": (
                    confusion_counts(
                        shared_errors,
                        prediction_column=("textcnn_prediction"),
                    )
                ),
            },
            "three_way_disagreement": {
                "count": int(len(disagreements)),
                "rate": rate(
                    len(disagreements),
                    total,
                ),
                "by_true_class": (
                    class_counts(disagreements).to_dict(orient="records")
                ),
            },
        },
    }

    summary_path = output_dir / "dair_v001_behavior_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    print(f"[RESULT] TextCNN fixes baseline: {len(textcnn_fixes)}")

    print(f"[RESULT] Baseline beats TextCNN: {len(textcnn_regressions)}")

    print(f"[RESULT] Shared wrong agreements: {len(shared_errors)}")

    print(f"[RESULT] Three-way disagreements: {len(disagreements)}")

    print(f"[OK] Behavior summary: {summary_path}")


if __name__ == "__main__":
    main()
