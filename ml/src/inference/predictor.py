from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from ml.src.interpretation.textcnn_interpreter import (
    interpret_textcnn_input,
    load_textcnn_checkpoint,
)
from ml.src.preprocessing.sequence import encode_text
from ml.src.preprocessing.tokenizer import tokenize
from ml.src.preprocessing.vocab import Vocabulary


LABELS = [
    "sadness",
    "joy",
    "love",
    "anger",
    "fear",
    "surprise",
]


class TextCNNPredictor:
    def __init__(
        self,
        config_path: str | Path,
        *,
        device: torch.device | None = None,
    ) -> None:
        self.config_path = Path(config_path)

        if not self.config_path.exists():
            raise FileNotFoundError(f"TextCNN config not found: {self.config_path}")

        with self.config_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            self.config: dict[str, Any] = yaml.safe_load(handle)

        self.model_id = str(self.config["experiment_id"])

        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        preprocessing = self.config["preprocessing"]

        self.lowercase = bool(preprocessing["lowercase"])

        self.max_sequence_length = int(preprocessing["max_sequence_length"])

        self.vocab = Vocabulary.load(preprocessing["vocab_path"])

        self.model, self.checkpoint = load_textcnn_checkpoint(
            self.config["artifacts"]["model"],
            device=self.device,
        )

    def _prepare_input(
        self,
        text: str,
    ) -> tuple[list[str], torch.Tensor]:
        stripped = text.strip()

        if not stripped:
            raise ValueError("text must not be empty")

        tokens = tokenize(
            stripped,
            lowercase=self.lowercase,
        )

        tokens = tokens[: self.max_sequence_length]

        input_ids = encode_text(
            tokens,
            self.vocab,
            max_length=(self.max_sequence_length),
        )

        tensor = torch.tensor(
            [input_ids],
            dtype=torch.long,
            device=self.device,
        )

        return tokens, tensor

    @torch.no_grad()
    def predict(
        self,
        text: str,
        *,
        include_distribution: bool = False,
    ) -> dict:
        tokens, input_ids = self._prepare_input(text)

        logits = self.model(input_ids)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

        values = probabilities.detach().cpu().tolist()

        ranked = sorted(
            [
                {
                    "label": label,
                    "probability": float(values[index]),
                }
                for index, label in enumerate(LABELS)
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )

        result = {
            "mode": "single_label",
            "model_id": self.model_id,
            "top_prediction": ranked[0],
            "top_3": ranked[:3],
            "input": {
                "text": text,
                "tokens": tokens,
                "token_count": len(tokens),
                "truncated": (
                    len(
                        tokenize(
                            text.strip(),
                            lowercase=self.lowercase,
                        )
                    )
                    > self.max_sequence_length
                ),
            },
            "probability_note": (
                "Softmax probabilities are not calibrated confidence estimates."
            ),
        }

        if include_distribution:
            result["distribution"] = ranked

        return result

    @torch.no_grad()
    def interpret(
        self,
        text: str,
        *,
        top_k: int = 10,
    ) -> dict:
        if top_k < 1 or top_k > 50:
            raise ValueError("top_k must be between 1 and 50")

        tokens, input_ids = self._prepare_input(text)

        kernel_sizes = [int(value) for value in self.checkpoint["kernel_sizes"]]

        result = interpret_textcnn_input(
            self.model,
            input_ids=input_ids,
            tokens=tokens,
            kernel_sizes=kernel_sizes,
            labels=LABELS,
            top_k=top_k,
        )

        return {
            "model_id": self.model_id,
            "text": text,
            "tokens": tokens,
            "prediction": (result["prediction"]),
            "scores": result["scores"],
            "top_text_spans": (result["top_text_spans"]),
            "padding_activation_count": (result["padding_activation_count"]),
            "interpretation_note": (
                "Highlighted spans are high-activation "
                "TextCNN convolution regions. They are "
                "not causal explanations of emotion or "
                "calibrated class importance."
            ),
        }

    def predict_batch(
        self,
        texts: list[str],
        *,
        include_distribution: bool = False,
    ) -> list[dict]:
        return [
            self.predict(
                text,
                include_distribution=(include_distribution),
            )
            for text in texts
        ]
