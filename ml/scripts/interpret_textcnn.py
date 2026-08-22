from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from ml.src.interpretation.textcnn_interpreter import (
    interpret_textcnn_input,
    load_textcnn_checkpoint,
)
from ml.src.preprocessing.sequence import (
    encode_text,
)
from ml.src.preprocessing.tokenizer import (
    tokenize,
)
from ml.src.preprocessing.vocab import (
    Vocabulary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Inspect TextCNN convolutional activation spans.")
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ml/configs/textcnn_dair_v001.yaml"),
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        return yaml.safe_load(handle)


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing = config["preprocessing"]

    vocab = Vocabulary.load(preprocessing["vocab_path"])

    model, checkpoint = load_textcnn_checkpoint(
        config["artifacts"]["model"],
        device=device,
    )

    tokens = tokenize(
        args.text,
        lowercase=bool(preprocessing["lowercase"]),
    )

    max_length = int(preprocessing["max_sequence_length"])

    truncated_tokens = tokens[:max_length]

    input_ids = encode_text(
        truncated_tokens,
        vocab,
        max_length=max_length,
    )

    tensor = torch.tensor(
        [input_ids],
        dtype=torch.long,
        device=device,
    )

    labels = [
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise",
    ]

    result = interpret_textcnn_input(
        model,
        input_ids=tensor,
        tokens=truncated_tokens,
        kernel_sizes=[int(value) for value in checkpoint["kernel_sizes"]],
        labels=labels,
        top_k=args.top_k,
    )

    result["text"] = args.text
    result["tokens"] = truncated_tokens
    result["model_id"] = config["experiment_id"]

    print(
        "[RESULT] Prediction: "
        f"{result['prediction']['label']} "
        f"({result['prediction']['score']:.4f})"
    )

    print("[RESULT] Top valid activation spans:")

    for index, span in enumerate(
        result["top_text_spans"],
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"kernel={span['kernel_size']} "
            f"filter={span['filter_index']} "
            f"activation="
            f"{span['activation']:.4f} "
            f"tokens="
            f"[{span['token_start']}:"
            f"{span['token_end_exclusive']}] "
            f"'{span['span_text']}'"
        )

    print(
        "[RESULT] Filters whose raw "
        "max overlaps padding: "
        f"{result['padding_activation_count']}"
    )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with args.output.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                result,
                handle,
                indent=2,
                ensure_ascii=False,
            )

        print(f"[OK] Interpretation saved: {args.output}")


if __name__ == "__main__":
    main()
