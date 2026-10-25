import hashlib
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.multiclass import OneVsRestClassifier


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "tfidf_goemotions_multilabel_v001"

CONFIG_PATH = ROOT / "ml/configs" / f"{EXPERIMENT}.json"
DATA_DIR = ROOT / "data/processed/goemotions_v001"
MODEL_DIR = ROOT / "ml/artifacts/models" / EXPERIMENT
METRICS_DIR = ROOT / "ml/artifacts/metrics"
PREDICTIONS_DIR = ROOT / "ml/artifacts/predictions"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(name, expected_count, expected_hash):
    path = DATA_DIR / f"{name}.jsonl"

    if sha256_file(path) != expected_hash:
        raise ValueError(f"{name}: checksum mismatch")

    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]

    if len(rows) != expected_count:
        raise ValueError(f"{name}: row count mismatch")

    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name}: duplicate example IDs")

    for row in rows:
        if not row["example_id"].startswith(f"goe_{name}_"):
            raise ValueError(f"{name}: unexpected example ID")

        if (
            hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            != row["text_sha256"]
        ):
            raise ValueError(f"{name}: text hash mismatch")

        labels = row["labels"]
        if (
            not isinstance(labels, list)
            or not labels
            or len(labels) != len(set(labels))
            or any(type(x) is not int or not 0 <= x < 28 for x in labels)
        ):
            raise ValueError(f"{name}: invalid labels")

    return rows


def make_targets(rows, num_labels):
    targets = np.zeros((len(rows), num_labels), dtype=np.int8)

    for i, row in enumerate(rows):
        targets[i, row["labels"]] = 1

    return targets


def choose_threshold(y_true, probabilities, candidates):
    results = []

    for threshold in candidates:
        predictions = (probabilities >= threshold).astype(np.int8)

        score = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        results.append(
            {
                "threshold": float(threshold),
                "macro_f1": float(score),
            }
        )

    best = max(
        results,
        key=lambda row: (
            row["macro_f1"],
            row["threshold"],
        ),
    )

    return best["threshold"], results


def evaluate(y_true, probabilities, threshold, label_names):
    predictions = (probabilities >= threshold).astype(np.int8)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        average=None,
        zero_division=0,
    )

    per_label_ap = average_precision_score(
        y_true,
        probabilities,
        average=None,
    )

    return {
        "threshold": float(threshold),
        "macro_f1": float(
            f1_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "micro_f1": float(
            f1_score(y_true, predictions, average="micro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "macro_average_precision": float(
            average_precision_score(y_true, probabilities, average="macro")
        ),
        "micro_average_precision": float(
            average_precision_score(y_true, probabilities, average="micro")
        ),
        "per_label": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
                "average_precision": float(per_label_ap[i]),
            }
            for i, name in enumerate(label_names)
        },
    }


def main():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    manifest_path = DATA_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if config["experiment_id"] != EXPERIMENT:
        raise ValueError("Experiment ID mismatch")

    if config["dataset_version"] != manifest["dataset_version"]:
        raise ValueError("Dataset version mismatch")

    label_names = manifest["label_names"]
    if len(label_names) != 28:
        raise ValueError("Expected 28 labels")

    train_info = manifest["processed_splits"]["train"]
    val_info = manifest["processed_splits"]["validation"]

    train = load_split("train", train_info["rows"], train_info["sha256"])
    validation = load_split("validation", val_info["rows"], val_info["sha256"])

    y_train = make_targets(train, len(label_names))
    y_val = make_targets(validation, len(label_names))

    if np.any(y_train.sum(axis=0) == 0):
        raise ValueError("Some labels have no training positives")

    print(f"Training examples: {len(train)}")
    print(f"Validation examples: {len(validation)}")

    vectorizer_config = dict(config["vectorizer"])
    vectorizer_config["ngram_range"] = tuple(vectorizer_config["ngram_range"])

    vectorizer = TfidfVectorizer(**vectorizer_config)

    print("Fitting TF-IDF...")
    x_train = vectorizer.fit_transform([row["text"] for row in train])
    x_val = vectorizer.transform([row["text"] for row in validation])

    print(f"Vocabulary features: {x_train.shape[1]}")

    classifier = OneVsRestClassifier(
        LogisticRegression(
            **config["classifier"],
            random_state=config["random_seed"],
        ),
        n_jobs=1,
    )

    print("Training 28 binary classifiers...")

    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        classifier.fit(x_train, y_train)

    probabilities = classifier.predict_proba(x_val)

    if probabilities.shape != y_val.shape:
        raise ValueError("Probability shape mismatch")

    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
    ):
        raise ValueError("Invalid model probabilities")

    threshold, search_results = choose_threshold(
        y_val,
        probabilities,
        config["threshold_candidates"],
    )

    metrics = evaluate(y_val, probabilities, threshold, label_names)

    metrics["experiment_id"] = EXPERIMENT
    metrics["dataset_version"] = config["dataset_version"]
    metrics["dataset_manifest_sha256"] = sha256_file(manifest_path)
    metrics["config_sha256"] = sha256_file(CONFIG_PATH)
    metrics["threshold_search"] = search_results
    metrics["evaluated_split"] = "validation"
    metrics["test_evaluated"] = False

    output_paths = [
        MODEL_DIR / "model.joblib",
        MODEL_DIR / "vectorizer.joblib",
        METRICS_DIR / f"{EXPERIMENT}_validation.json",
        PREDICTIONS_DIR / f"{EXPERIMENT}_validation.jsonl",
    ]

    if any(path.exists() for path in output_paths):
        raise FileExistsError(
            "Experiment artifacts already exist. "
            "Use a new experiment version instead of overwriting."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(classifier, output_paths[0])
    joblib.dump(vectorizer, output_paths[1])

    output_paths[2].write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    predictions = (probabilities >= threshold).astype(np.int8)

    with output_paths[3].open("x", encoding="utf-8", newline="\n") as stream:
        for i, row in enumerate(validation):
            record = {
                "example_id": row["example_id"],
                "source_id": row["source_id"],
                "true_labels": row["labels"],
                "predicted_labels": np.flatnonzero(predictions[i]).tolist(),
                "probabilities": probabilities[i].tolist(),
            }
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    print(f"Selected threshold: {threshold:.2f}")
    print(f"Validation macro F1: {metrics['macro_f1']:.4f}")
    print(f"Validation micro F1: {metrics['micro_f1']:.4f}")
    print(f"Validation macro AP: {metrics['macro_average_precision']:.4f}")
    print("[OK] Baseline training completed")


if __name__ == "__main__":
    main()
