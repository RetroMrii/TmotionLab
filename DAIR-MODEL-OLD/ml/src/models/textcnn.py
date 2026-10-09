from __future__ import annotations

import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int,
        embedding_dim: int,
        num_filters: int,
        kernel_sizes: list[int],
        num_classes: int,
        dropout: float,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()

        if vocab_size < 2:
            raise ValueError("vocab_size must be >= 2")

        if embedding_dim < 1:
            raise ValueError("embedding_dim must be >= 1")

        if num_filters < 1:
            raise ValueError("num_filters must be >= 1")

        if not kernel_sizes:
            raise ValueError("kernel_sizes cannot be empty")

        if any(size < 1 for size in kernel_sizes):
            raise ValueError("kernel sizes must be >= 1")

        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.convolutions = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=kernel_size,
                )
                for kernel_size in kernel_sizes
            ]
        )

        self.dropout = nn.Dropout(dropout)

        classifier_input_dim = num_filters * len(kernel_sizes)

        self.classifier = nn.Linear(
            classifier_input_dim,
            num_classes,
        )

    def extract_conv_activations(
        self,
        input_ids: torch.Tensor,
    ) -> list[torch.Tensor]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length]")

        embedded = self.embedding(input_ids)

        # [batch, sequence, embedding]
        # ->
        # [batch, embedding, sequence]
        embedded = embedded.transpose(1, 2)

        return [torch.relu(convolution(embedded)) for convolution in self.convolutions]

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length]")

        embedded = self.embedding(input_ids)

        # [batch, sequence, embedding]
        # ->
        # [batch, embedding, sequence]
        embedded = embedded.transpose(1, 2)

        pooled_outputs = []

        for convolution in self.convolutions:
            features = torch.relu(convolution(embedded))

            pooled = torch.max(
                features,
                dim=2,
            ).values

            pooled_outputs.append(pooled)

        combined = torch.cat(
            pooled_outputs,
            dim=1,
        )

        combined = self.dropout(combined)

        logits = self.classifier(combined)

        return logits
