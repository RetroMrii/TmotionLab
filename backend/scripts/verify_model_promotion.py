"""Verify saved validation evidence; never train or read the held-out test split.

--write publishes aggregate evidence and a serving manifest after verification.
Without it, compare the derived evidence to the checked-in snapshots.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "roberta_goemotions_synthetic10_v001"
RUN = "ml/artifacts/optimization/opt010/synthetic10"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect():
    hashes = {}

    def read(name):
        path = ROOT / name
        hashes[name] = digest(path)
        return json.loads(path.read_text(encoding="utf-8"))

    manifest = read("data/processed/goemotions_v001/manifest.json")
    labels = manifest["label_names"]
    config = read(f"{RUN}/best_model/config.json")
    assert len(labels) == len(set(labels)) == 28
    assert config["id2label"] == {str(i): label for i, label in enumerate(labels)}
    assert config["label2id"] == {label: i for i, label in enumerate(labels)}
    assert config["problem_type"] == "multi_label_classification"
    provenance = read(f"{RUN}/provenance.json")
    training_config = read("ml/configs/roberta_goemotions_multilabel_real_v001.json")
    assert training_config["max_length"] == 48
    assert provenance["config_sha256"] == hashes["ml/configs/roberta_goemotions_multilabel_real_v001.json"]
    assert provenance["real_manifest_sha256"] == hashes["data/processed/goemotions_v001/manifest.json"]
    assert provenance["synthetic_fraction"] == .1 and provenance["threshold"] == .25
    assert provenance["seed"] == 42 and provenance["test_evaluated"] is False
    synthetic = read("data/processed/synthetic_goemotions_opt009_v001/manifest.json")
    assert provenance["synthetic_manifest_sha256"] == hashes["data/processed/synthetic_goemotions_opt009_v001/manifest.json"]
    assert digest(ROOT / "data/processed/synthetic_goemotions_opt009_v001/train.jsonl") == synthetic["training_data_sha256"] == provenance["synthetic_data_sha256"]
    validation = ROOT / "data/processed/goemotions_v001/validation.jsonl"
    hashes[validation.relative_to(ROOT).as_posix()] = digest(validation)
    rows = [json.loads(line) for line in validation.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 5426 and len({row["example_id"] for row in rows}) == 5426
    gold = np.zeros((5426, 28), dtype=int)
    for i, row in enumerate(rows):
        gold[i, row["labels"]] = 1
    runs = []
    for experiment, modes in [("opt007", ["baseline", "weighted"]), ("opt010", ["control", "synthetic10", "synthetic25", "synthetic40"]), ("opt012", ["control", "synthetic10"])]:
        for mode in modes:
            folder = f"ml/artifacts/optimization/{experiment}/{mode}"
            metrics = read(f"{folder}/validation_metrics.json")
            summary = read(f"{folder}/training_summary.json")
            path = ROOT / folder / "validation_predictions.jsonl"
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
            predictions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            assert len(predictions) == len(rows)
            for actual, saved in zip(rows, predictions, strict=True):
                assert actual["example_id"] == saved["example_id"]
                assert sorted(actual["labels"]) == sorted(saved["true_labels"])
            scores = np.asarray([row["probabilities"] for row in predictions])
            assert scores.shape == (5426, 28) and np.isfinite(scores).all()
            assert ((scores >= 0) & (scores <= 1)).all()
            computed = {"macro_average_precision": average_precision_score(gold, scores, average="macro"),
                        "macro_f1": f1_score(gold, scores >= .25, average="macro", zero_division=0),
                        "micro_f1": f1_score(gold, scores >= .25, average="micro", zero_division=0)}
            for key, value in computed.items():
                assert abs(value - metrics[key]) < 1e-10, (experiment, mode, key)
            ap = average_precision_score(gold, scores, average=None)
            for i, label in enumerate(labels):
                assert abs(ap[i] - metrics["per_label"][label]["average_precision"]) < 1e-10
                assert int(gold[:, i].sum()) == metrics["per_label"][label]["support"]
            if experiment in {"opt010", "opt012"}:
                assert summary["optimizer_steps"] == 5408
                assert summary["training_metrics"]["epoch"] == 4
            runs.append({"experiment": experiment.upper(), "mode": mode, **computed, "per_label": metrics["per_label"]})
    opt008 = read("reports/goemotions/optimization/opt008_weighted_comparison.json")
    read("reports/goemotions/opt011/analysis.json")
    selected = next(row for row in runs if row["experiment"] == "OPT010" and row["mode"] == "synthetic10")
    control = next(row for row in runs if row["experiment"] == "OPT010" and row["mode"] == "control")
    deltas = [{"label": label, "support": selected["per_label"][label]["support"],
               "delta_ap": selected["per_label"][label]["average_precision"] - control["per_label"][label]["average_precision"]} for label in labels]
    evidence = {"model_id": MODEL_ID, "split": "validation", "examples": 5426, "seed": 42,
                "threshold": .25, "runs": runs, "deltas": deltas,
                "opt008": {name: {key: values[key] for key in ("fixed_025", "out_of_fold", "macro_ap")} for name, values in opt008["results"].items()},
                "synthetic": {key: synthetic[key] for key in ("source_candidates", "retained", "excluded", "exclusion_reasons", "human_review")},
                "source_sha256": hashes, "independent_test_evaluated": False}
    serving = {"primary_experiment": MODEL_ID, "source_experiment": "OPT-010 synthetic10",
               "model_directory": f"{RUN}/best_model", "seed": 42, "classification_threshold": .25,
               "threshold_selection_split": "validation", "threshold_policy": "fixed_global_inherited_0.25",
               "dataset_version": "goemotions_v001", "test_time_tuning_allowed": False,
               "selection_metric": "validation_macro_average_precision", "independent_test_evaluated": False,
               "dataset_manifest_sha256": hashes["data/processed/goemotions_v001/manifest.json"],
               "validation_predictions": f"{RUN}/validation_predictions.jsonl",
               "validation_predictions_sha256": hashes[f"{RUN}/validation_predictions.jsonl"],
               "validation_metrics": {key: selected[key] for key in ("macro_average_precision", "macro_f1", "micro_f1")},
               "model_file_hashes": {p.name: digest(p) for p in sorted((ROOT / RUN / "best_model").iterdir()) if p.is_file()}}
    return evidence, serving


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    evidence, serving = collect()
    for name, data in [("frontend/src/data/optimization.json", evidence), ("backend/model-manifests/goemotions_synthetic10_v001.json", serving)]:
        path = ROOT / name
        if args.write:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        else:
            assert json.loads(path.read_text(encoding="utf-8")) == data, f"Published evidence differs: {name}"
    print(json.dumps({"verified_validation_runs": len(evidence["runs"]), "examples_per_run": 5426,
                      "model_id": MODEL_ID, "test_split_read": False, "metrics": serving["validation_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
