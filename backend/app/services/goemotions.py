"""Read-only serving of the frozen P11 candidate; never imports archived ML code."""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from pathlib import Path
from threading import Lock

ROOT = Path(__file__).resolve().parents[3]
MODEL_ID = "roberta_goemotions_rare4x_v001"
MODEL_DIR = ROOT / "ml/artifacts/models" / MODEL_ID / "best_model"
FREEZE_PATH = ROOT / "reports/goemotions/final_model_freeze_v001.json"
MANIFEST_PATH = ROOT / "data/processed/goemotions_v001/manifest.json"
THRESHOLD = 0.25
MAX_TOKENS = 48
MAX_CHARS = 5000
NOTE = (
    "Independent sigmoid scores are not calibrated confidence and need not sum to one. "
    "A detection is a model output, not a person's psychological state. "
    "Neutral is learned and may co-occur with other labels."
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_artifacts(model_dir=MODEL_DIR, freeze_path=FREEZE_PATH, manifest_path=MANIFEST_PATH):
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    if (freeze["primary_experiment"] != MODEL_ID or freeze["seed"] != 42
            or freeze["classification_threshold"] != THRESHOLD
            or freeze["threshold_selection_split"] != "validation"
            or freeze["dataset_version"] != "goemotions_v001"
            or freeze["test_time_tuning_allowed"] is not False):
        raise ValueError("Unexpected frozen model/threshold provenance")
    expected = freeze["model_file_hashes"]
    required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
    if not required.issubset(expected):
        raise ValueError("Incomplete frozen file hashes")
    actual = {p.name: sha256(p) for p in model_dir.iterdir() if p.is_file()}
    if actual != expected:
        raise ValueError("Frozen model file SHA-256 mismatch")
    if sha256(manifest_path) != freeze["dataset_manifest_sha256"]:
        raise ValueError("Frozen dataset manifest SHA-256 mismatch")
    labels = json.loads(manifest_path.read_text(encoding="utf-8"))["label_names"]
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if (len(labels) != 28 or len(set(labels)) != 28
            or config.get("id2label") != {str(i): label for i, label in enumerate(labels)}
            or config.get("label2id") != {label: i for i, label in enumerate(labels)}):
        raise ValueError("Model label mapping differs from processed manifest")
    return labels


def score_payload(labels: list[str], probabilities: list[float]) -> dict:
    if len(labels) != 28 or len(probabilities) != 28:
        raise ValueError("Expected exactly 28 model scores")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("Invalid model scores")
    scores = [dict(label_id=i, label=label, score=float(p))
              for i, (label, p) in enumerate(zip(labels, probabilities))]
    ranked = sorted(scores, key=lambda item: (-item["score"], item["label_id"]))
    return {
        "scores": scores,
        "detected_labels": [item["label"] for item in scores if item["score"] >= THRESHOLD],
        "ranked_scores": ranked[:5],
    }


class GoEmotionsPredictor:
    model_id = MODEL_ID

    def __init__(self):
        # Hashes are checked before Transformers can load any checkpoint.
        self.labels = verify_artifacts()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._lock = Lock()
        requested = os.getenv("TMOTIONLAB_DEVICE", "cpu")
        if requested not in {"cpu", "cuda", "auto"}:
            raise ValueError("TMOTIONLAB_DEVICE must be cpu, cuda or auto")
        self.notes = []
        self.device = "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"
        if requested == "cuda" and self.device == "cpu":
            self.notes.append("CUDA unavailable; using CPU.")
        torch.use_deterministic_algorithms(True)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True,
        )
        if self.model.config.num_labels != 28 or self.model.config.id2label != dict(enumerate(self.labels)):
            raise ValueError("Loaded classifier has an invalid label mapping")
        self.model.eval()
        try:
            self.model.to(self.device)
            self.predict("Model availability check.")
        except RuntimeError:
            if self.device == "cpu":
                raise
            logging.getLogger(__name__).exception("CUDA failed; retrying on CPU")
            self.device = "cpu"
            self.model.to("cpu")
            self.notes.append("CUDA initialization/inference failed; using CPU.")
            self.predict("Model availability check.")

    def predict(self, text: str) -> dict:
        import torch

        if not isinstance(text, str) or not text.strip() or len(text) > MAX_CHARS:
            raise ValueError("text must contain 1–5000 characters and not be blank")
        with self._lock, torch.inference_mode():
            original_count = len(self.tokenizer(text, truncation=False)["input_ids"])
            encoded = self.tokenizer(
                text, padding=True, truncation=True, max_length=MAX_TOKENS, return_tensors="pt",
            )
            ids = encoded["input_ids"][0].tolist()
            logits = self.model(**{k: v.to(self.device) for k, v in encoded.items()}).logits
            if tuple(logits.shape) != (1, 28) or not torch.isfinite(logits).all():
                raise ValueError("Invalid classifier logits")
            scores = score_payload(self.labels, torch.sigmoid(logits)[0].cpu().tolist())
        return {
            "schema_version": "1", "task": "multi_label", "model_id": MODEL_ID,
            "dataset": "goemotions_v001", "seed": 42, "threshold": THRESHOLD,
            "score_type": "sigmoid", "device": self.device,
            "input": {
                "text": text, "tokens": self.tokenizer.convert_ids_to_tokens(ids),
                "token_count": len(ids), "original_token_count": original_count,
                "max_tokens": MAX_TOKENS, "truncated": original_count > MAX_TOKENS,
            },
            **scores, "score_note": NOTE, "availability_notes": self.notes,
        }
