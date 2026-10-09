from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ml.src.models.textcnn import TextCNN


def load_textcnn_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> tuple[TextCNN, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"TextCNN checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    required_keys = {
        "model_state_dict",
        "vocab_size",
        "embedding_dim",
        "num_filters",
        "kernel_sizes",
        "num_classes",
        "dropout",
        "max_sequence_length",
    }

    missing = required_keys - set(checkpoint)

    if missing:
        raise ValueError(f"Checkpoint missing required keys: {sorted(missing)}")

    model = TextCNN(
        vocab_size=int(checkpoint["vocab_size"]),
        embedding_dim=int(checkpoint["embedding_dim"]),
        num_filters=int(checkpoint["num_filters"]),
        kernel_sizes=[int(value) for value in checkpoint["kernel_sizes"]],
        num_classes=int(checkpoint["num_classes"]),
        dropout=float(checkpoint["dropout"]),
        padding_idx=0,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def interpret_textcnn_input(
    model: TextCNN,
    *,
    input_ids: torch.Tensor,
    tokens: list[str],
    kernel_sizes: list[int],
    labels: list[str],
    top_k: int = 10,
) -> dict:
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch_size, sequence_length]")

    if input_ids.shape[0] != 1:
        raise ValueError("Interpretation currently supports one example at a time")

    if len(kernel_sizes) != len(model.convolutions):
        raise ValueError("kernel_sizes does not match model convolutions")

    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    logits = model(input_ids)

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    prediction_id = int(
        torch.argmax(
            probabilities,
            dim=1,
        ).item()
    )

    conv_activations = model.extract_conv_activations(input_ids)

    records: list[dict] = []

    valid_token_count = len(tokens)

    for kernel_index, (
        kernel_size,
        activations,
    ) in enumerate(
        zip(
            kernel_sizes,
            conv_activations,
        )
    ):
        # activations:
        # [1, num_filters, num_positions]

        filter_max_values, filter_max_positions = torch.max(
            activations[0],
            dim=1,
        )

        for filter_index in range(activations.shape[1]):
            start = int(filter_max_positions[filter_index].item())

            end = start + int(kernel_size)

            activation = float(filter_max_values[filter_index].item())

            contains_padding = end > valid_token_count

            available_end = min(
                end,
                valid_token_count,
            )

            span_tokens = (
                tokens[start:available_end] if start < valid_token_count else []
            )

            records.append(
                {
                    "kernel_index": (kernel_index),
                    "kernel_size": int(kernel_size),
                    "filter_index": int(filter_index),
                    "activation": activation,
                    "token_start": start,
                    "token_end_exclusive": end,
                    "span_tokens": span_tokens,
                    "span_text": " ".join(span_tokens),
                    "contains_padding": (contains_padding),
                    "valid_text_span": bool(span_tokens) and not contains_padding,
                }
            )

    records.sort(
        key=lambda item: item["activation"],
        reverse=True,
    )

    top_all = records[:top_k]

    valid_records = [record for record in records if record["valid_text_span"]]

    top_text_spans = valid_records[:top_k]

    probability_values = probabilities[0].cpu().tolist()

    scores = [
        {
            "label": labels[index],
            "score": float(score),
        }
        for index, score in enumerate(probability_values)
    ]

    scores.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "prediction": {
            "label_id": prediction_id,
            "label": labels[prediction_id],
            "score": float(
                probabilities[
                    0,
                    prediction_id,
                ].item()
            ),
        },
        "scores": scores,
        "token_count": valid_token_count,
        "top_activations": top_all,
        "top_text_spans": top_text_spans,
        "padding_activation_count": sum(
            1 for record in records if record["contains_padding"]
        ),
    }
