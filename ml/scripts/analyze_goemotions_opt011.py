import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import KFold

from ml.src.data.goemotions_dataset import load_goemotions_rows


ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = ROOT / "ml/artifacts/optimization/opt010"
OUTPUT = ROOT / "reports/goemotions/opt011"

MODES = ("control", "synthetic10")
SEED = 42
FOLDS = 5
FIXED_THRESHOLD = 0.25
THRESHOLD_GRID = np.round(np.arange(0.05, 0.9501, 0.025), 6)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            data,
            stream,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")


def target_matrix(rows, num_labels):
    targets = np.zeros((len(rows), num_labels), dtype=np.int8)
    for i, row in enumerate(rows):
        for label in row["labels"]:
            targets[i, label] = 1
    return targets


def load_predictions(mode, validation_rows):
    run_dir = INPUT_ROOT / mode
    path = run_dir / "validation_predictions.jsonl"
    metrics_path = run_dir / "validation_metrics.json"
    provenance_path = run_dir / "provenance.json"

    records = read_jsonl(path)
    metrics = read_json(metrics_path)
    provenance = read_json(provenance_path)

    if len(records) != len(validation_rows):
        raise ValueError(f"{mode}: validation count mismatch")

    if (
        provenance["experiment"] != "OPT-010"
        or provenance["mode"] != mode
        or provenance["seed"] != SEED
        or provenance["expected_optimizer_steps"] != 5408
    ):
        raise ValueError(f"{mode}: provenance mismatch")

    if (
        metrics["optimizer_steps"] != 5408
        or metrics["evaluation_split"] != "validation"
        or metrics["test_evaluated"] is not False
    ):
        raise ValueError(f"{mode}: metrics mismatch")

    expected_fraction = 0.0 if mode == "control" else 0.10
    if abs(provenance["synthetic_fraction"] - expected_fraction) > 1e-12:
        raise ValueError(f"{mode}: synthetic fraction mismatch")

    probabilities = np.empty((len(records), 28), dtype=np.float64)

    seen = set()

    for i, (record, row) in enumerate(zip(records, validation_rows)):
        identifier = record["example_id"]

        if identifier in seen:
            raise ValueError(f"{mode}: duplicate validation ID")
        seen.add(identifier)

        if identifier != row["example_id"]:
            raise ValueError(f"{mode}: validation ordering mismatch at row {i}")

        if sorted(record["true_labels"]) != sorted(row["labels"]):
            raise ValueError(f"{mode}: validation ground truth mismatch at row {i}")

        values = np.asarray(record["probabilities"], dtype=np.float64)

        if values.shape != (28,):
            raise ValueError(f"{mode}: invalid probability vector")

        probabilities[i] = values

    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError(f"{mode}: invalid probabilities")

    return {
        "probabilities": probabilities,
        "metrics": metrics,
        "provenance": provenance,
        "hashes": {
            "predictions_sha256": sha256_file(path),
            "metrics_sha256": sha256_file(metrics_path),
            "provenance_sha256": sha256_file(provenance_path),
        },
    }


def score_summary(y, p, predictions):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        macro_ap = average_precision_score(y, p, average="macro")

    return {
        "macro_ap": float(macro_ap),
        "macro_f1": float(
            f1_score(
                y,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "micro_f1": float(
            f1_score(
                y,
                predictions,
                average="micro",
                zero_division=0,
            )
        ),
    }


def per_label_metrics(y, p, predictions, names):
    output = []

    for j, name in enumerate(names):
        actual = y[:, j].astype(bool)
        predicted = predictions[:, j].astype(bool)

        tp = int(np.sum(actual & predicted))
        fp = int(np.sum(~actual & predicted))
        fn = int(np.sum(actual & ~predicted))

        precision, recall, f1, _ = precision_recall_fscore_support(
            actual,
            predicted,
            average="binary",
            zero_division=0,
        )

        ap = float(average_precision_score(actual, p[:, j]))

        output.append(
            {
                "label": name,
                "support": int(actual.sum()),
                "ap": ap,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    return output


def choose_threshold(y, p):
    """
    Select threshold by training-fold binary F1.

    In ties, prefer closest to 0.25, then lower.
    This is deterministic and uses no held-out rows.
    """
    best_threshold = None
    best_key = None

    for threshold in THRESHOLD_GRID:
        predictions = p >= threshold
        score = f1_score(y, predictions, zero_division=0)

        key = (
            float(score),
            -abs(float(threshold) - FIXED_THRESHOLD),
            -float(threshold),
        )

        if best_key is None or key > best_key:
            best_key = key
            best_threshold = float(threshold)

    return best_threshold


def crossfit_thresholds(y, p, splits):
    """
    Every row receives predictions from a threshold
    tuned without that row's fold.

    Model probabilities were already produced by a
    validation-selected checkpoint. This is therefore
    threshold cross-fitting, not fully independent
    model evaluation.
    """
    predictions = np.zeros_like(y, dtype=bool)
    thresholds_by_fold = []

    for fold, (fit_idx, eval_idx) in enumerate(splits):
        if np.intersect1d(fit_idx, eval_idx).size:
            raise RuntimeError("Fold overlap")

        thresholds = np.empty(y.shape[1], dtype=np.float64)

        for j in range(y.shape[1]):
            threshold = choose_threshold(
                y[fit_idx, j],
                p[fit_idx, j],
            )
            thresholds[j] = threshold

            predictions[eval_idx, j] = p[eval_idx, j] >= threshold

        thresholds_by_fold.append(
            {
                "fold": fold,
                "fit_count": len(fit_idx),
                "evaluation_count": len(eval_idx),
                "thresholds": thresholds.tolist(),
            }
        )

    return predictions, thresholds_by_fold


def difference_rows(control, synthetic):
    output = []

    if len(control) != len(synthetic):
        raise ValueError("Per-label metrics count mismatch")

    for left, right in zip(control, synthetic):
        if left["label"] != right["label"] or left["support"] != right["support"]:
            raise ValueError("Per-label alignment mismatch")

        output.append(
            {
                "label": left["label"],
                "support": left["support"],
                "control_ap": left["ap"],
                "synthetic_ap": right["ap"],
                "delta_ap": right["ap"] - left["ap"],
                "control_f1": left["f1"],
                "synthetic_f1": right["f1"],
                "delta_f1": right["f1"] - left["f1"],
                "control_precision": left["precision"],
                "synthetic_precision": right["precision"],
                "control_recall": left["recall"],
                "synthetic_recall": right["recall"],
                "control_tp": left["tp"],
                "synthetic_tp": right["tp"],
                "delta_fp": right["fp"] - left["fp"],
                "delta_fn": right["fn"] - left["fn"],
            }
        )

    return output


def format_summary(title, results):
    return (
        f"{title}: "
        f"AP={results['macro_ap']:.6f}, "
        f"macro F1={results['macro_f1']:.6f}, "
        f"micro F1={results['micro_f1']:.6f}"
    )


def markdown_table(rows, sort_key):
    ordered = sorted(
        rows,
        key=lambda row: row[sort_key],
        reverse=True,
    )

    lines = [
        "| Emotion | Support | Delta AP | Delta F1 | Delta FP | Delta FN |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in ordered:
        lines.append(
            f"| {row['label']} "
            f"| {row['support']} "
            f"| {row['delta_ap']:+.4f} "
            f"| {row['delta_f1']:+.4f} "
            f"| {row['delta_fp']:+d} "
            f"| {row['delta_fn']:+d} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="store_true",
        help="Write new report files; otherwise print preflight.",
    )
    args = parser.parse_args()

    validation, names = load_goemotions_rows("validation")

    if len(validation) != 5426 or len(names) != 28:
        raise ValueError("Unexpected validation dataset dimensions")

    y = target_matrix(validation, 28)

    if np.any(y.sum(axis=0) == 0):
        raise ValueError("At least one validation label has no positives")

    inputs = {mode: load_predictions(mode, validation) for mode in MODES}

    fixed = {}
    fixed_rows = {}

    for mode in MODES:
        p = inputs[mode]["probabilities"]
        predictions = p >= FIXED_THRESHOLD

        fixed[mode] = score_summary(y, p, predictions)
        fixed_rows[mode] = per_label_metrics(y, p, predictions, names)

        saved = inputs[mode]["metrics"]

        for key in (
            "macro_average_precision",
            "macro_f1",
            "micro_f1",
        ):
            calculated_key = "macro_ap" if key == "macro_average_precision" else key

            if abs(fixed[mode][calculated_key] - saved[key]) > 1e-6:
                raise ValueError(
                    f"{mode}: stored {key} disagrees with recomputed metric"
                )

    print("\nOPT-011 — PREFLIGHT")
    print("Validation examples:", len(validation))
    print("Labels:", len(names))
    print("Fixed threshold:", FIXED_THRESHOLD)
    print(format_summary("Control fixed", fixed["control"]))
    print(format_summary("Synthetic10 fixed", fixed["synthetic10"]))
    print("Prediction alignment: PASS")
    print("Saved metric reproduction: PASS")

    if not args.run:
        print("No files written.")
        return

    if OUTPUT.exists():
        raise FileExistsError(f"OPT-011 report directory exists: {OUTPUT}")

    splitter = KFold(
        n_splits=FOLDS,
        shuffle=True,
        random_state=SEED,
    )
    splits = list(splitter.split(y))

    oof = {}
    oof_rows = {}
    threshold_reports = {}

    for mode in MODES:
        p = inputs[mode]["probabilities"]

        predictions, fold_thresholds = crossfit_thresholds(y, p, splits)

        oof[mode] = score_summary(y, p, predictions)

        oof_rows[mode] = per_label_metrics(y, p, predictions, names)

        threshold_reports[mode] = [
            {
                **fold,
                "thresholds_by_label": dict(zip(names, fold["thresholds"])),
            }
            for fold in fold_thresholds
        ]

    fixed_deltas = difference_rows(
        fixed_rows["control"],
        fixed_rows["synthetic10"],
    )

    oof_deltas = difference_rows(
        oof_rows["control"],
        oof_rows["synthetic10"],
    )

    report = {
        "experiment": "OPT-011",
        "status": "analysis_complete",
        "seed": SEED,
        "validation_count": len(validation),
        "label_names": names,
        "fixed_threshold": FIXED_THRESHOLD,
        "folds": FOLDS,
        "threshold_grid": THRESHOLD_GRID.tolist(),
        "threshold_selection": ("Training-fold F1 maximization per label"),
        "fixed_metrics": fixed,
        "oof_threshold_metrics": oof,
        "fixed_per_label": fixed_deltas,
        "oof_per_label": oof_deltas,
        "fold_thresholds": threshold_reports,
        "source_hashes": {mode: inputs[mode]["hashes"] for mode in MODES},
        "limitations": [
            "Only one model-training seed.",
            "Validation checkpoints were selected using macro AP.",
            "OOF applies to threshold fitting only.",
            "KFold is not multilabel-stratified.",
            "No independent test-set evaluation.",
            "Thresholds optimized per label for F1, not micro F1.",
        ],
        "production_approved": False,
    }

    OUTPUT.mkdir(parents=True, exist_ok=False)

    save_json(OUTPUT / "analysis.json", report)

    markdown = [
        "# OPT-011 — Synthetic10 vs Control",
        "",
        "## Fixed 0.25 threshold",
        "",
        format_summary("Control", fixed["control"]),
        "",
        format_summary("Synthetic10", fixed["synthetic10"]),
        "",
        "## Five-fold out-of-fold threshold tuning",
        "",
        format_summary("Control", oof["control"]),
        "",
        format_summary("Synthetic10", oof["synthetic10"]),
        "",
        "## Fixed-threshold per-label comparison",
        "",
        markdown_table(fixed_deltas, "delta_ap"),
        "",
        "## OOF-threshold per-label comparison",
        "",
        markdown_table(oof_deltas, "delta_f1"),
        "",
        "## Methodological limits",
        "",
        "Thresholds are cross-fitted, but model checkpoint "
        "selection already used this validation dataset.",
        "This is not an independent generalization estimate.",
        "No production promotion is authorized.",
        "",
    ]

    with (OUTPUT / "report.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(markdown))

    print("\nOPT-011 — RESULTS")
    print(format_summary("Control OOF", oof["control"]))
    print(format_summary("Synthetic10 OOF", oof["synthetic10"]))

    print("\nTOP AP IMPROVEMENTS")
    for row in sorted(
        fixed_deltas,
        key=lambda row: row["delta_ap"],
        reverse=True,
    )[:10]:
        print(
            f"{row['label']:17s} "
            f"AP={row['delta_ap']:+.4f} "
            f"F1={row['delta_f1']:+.4f} "
            f"FP={row['delta_fp']:+d} "
            f"FN={row['delta_fn']:+d}"
        )

    print("\nTOP AP REGRESSIONS")
    for row in sorted(
        fixed_deltas,
        key=lambda row: row["delta_ap"],
    )[:10]:
        print(
            f"{row['label']:17s} "
            f"AP={row['delta_ap']:+.4f} "
            f"F1={row['delta_f1']:+.4f} "
            f"FP={row['delta_fp']:+d} "
            f"FN={row['delta_fn']:+d}"
        )

    print("\nSaved:", OUTPUT)
    print("Production approval: NOT GRANTED")


if __name__ == "__main__":
    main()
