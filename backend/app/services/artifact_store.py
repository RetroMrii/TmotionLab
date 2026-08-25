from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


COMPARISON_PATH = Path("ml/artifacts/comparisons/dair_v001_model_comparison.csv")

ERRORS_PATH = Path("ml/artifacts/errors/textcnn_dair_v001_errors.csv")

CONFUSIONS_PATH = Path("ml/artifacts/errors/textcnn_dair_v001_confusion_pairs.csv")

METRIC_PATHS = {
    "baseline_tfidf_dair_v001": Path(
        "ml/artifacts/metrics/baseline_tfidf_dair_v001_metrics.json"
    ),
    "numpy_mlp_dair_v002": Path(
        "ml/artifacts/metrics/numpy_mlp_dair_v002_metrics.json"
    ),
    "textcnn_dair_v001": Path("ml/artifacts/metrics/textcnn_dair_v001_metrics.json"),
}


def _load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    return pd.read_csv(path)


def _load_json(
    path: Path,
) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")

    return data


def _python_value(
    value: Any,
) -> Any:
    if value is None:
        return None

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False

    if (
        isinstance(
            missing,
            bool,
        )
        and missing
    ):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    return value


def _row_to_dict(
    row: pd.Series,
) -> dict:
    return {str(key): _python_value(value) for key, value in row.items()}


class ArtifactStore:
    def __init__(self) -> None:
        self.comparison = _load_csv(COMPARISON_PATH)

        self.errors = _load_csv(ERRORS_PATH)

        self.confusions = _load_csv(CONFUSIONS_PATH)

        self.metrics = {
            model_id: _load_json(path) for model_id, path in METRIC_PATHS.items()
        }

        self._validate()

    def _validate(self) -> None:
        if self.comparison["example_id"].duplicated().any():
            raise ValueError("Comparison artifact has duplicate example_id values")

        if self.errors["example_id"].duplicated().any():
            raise ValueError("Error artifact has duplicate example_id values")

    def get_metrics(
        self,
    ) -> list[dict]:
        output = []

        for model_id, payload in self.metrics.items():
            test = payload["test"]

            output.append(
                {
                    "model_id": model_id,
                    "accuracy": float(test["accuracy"]),
                    "macro_f1": float(test["macro_f1"]),
                    "weighted_f1": float(test["weighted_f1"]),
                }
            )

        return output

    def get_comparison(
        self,
        example_id: str,
    ) -> dict | None:
        matches = self.comparison[self.comparison["example_id"] == example_id]

        if matches.empty:
            return None

        return _row_to_dict(matches.iloc[0])

    def query_errors(
        self,
        *,
        true_label: str | None = None,
        predicted_label: str | None = None,
        min_score: float = 0.0,
        max_score: float = 1.0,
        shared_only: bool = False,
        baseline_regression_only: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> tuple[int, list[dict]]:
        frame = self.errors

        if true_label is not None:
            frame = frame[frame["true_label_name"] == true_label]

        if predicted_label is not None:
            frame = frame[frame["textcnn_prediction"] == predicted_label]

        frame = frame[
            (frame["textcnn_score"] >= min_score)
            & (frame["textcnn_score"] <= max_score)
        ]

        if shared_only:
            frame = frame[frame["all_agree_and_wrong"] == True]

        if baseline_regression_only:
            frame = frame[frame["baseline_correct_textcnn_wrong"] == True]

        total = len(frame)

        page = frame.iloc[offset : offset + limit]

        return (
            total,
            [_row_to_dict(row) for _, row in page.iterrows()],
        )

    def get_confusions(
        self,
        *,
        limit: int = 20,
    ) -> list[dict]:
        frame = self.confusions.head(limit)

        return [_row_to_dict(row) for _, row in frame.iterrows()]
