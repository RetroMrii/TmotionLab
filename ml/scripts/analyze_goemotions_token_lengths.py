import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed" / "goemotions_v001"
MANIFEST_PATH = DATA_DIR / "manifest.json"

REPORT_PATH = ROOT / "reports" / "goemotions" / "roberta_token_lengths_v001.json"

MODEL_ID = "FacebookAI/roberta-base"
SPLITS = ("train", "validation")
BATCH_SIZE = 512
CANDIDATE_LENGTHS = (32, 48, 64, 96, 128, 192, 256)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_texts(split: str, manifest: dict) -> list[str]:
    info = manifest["processed_splits"][split]
    path = DATA_DIR / f"{split}.jsonl"

    if sha256_file(path) != info["sha256"]:
        raise ValueError(f"{split}: checksum mismatch")

    texts = []
    seen_ids = set()

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue

            row = json.loads(line)

            if row["example_id"] in seen_ids:
                raise ValueError(f"{split}: duplicate example ID")

            seen_ids.add(row["example_id"])

            text = row["text"]

            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{split}: invalid text")

            if hashlib.sha256(text.encode("utf-8")).hexdigest() != row["text_sha256"]:
                raise ValueError(f"{split}: text hash mismatch")

            texts.append(text)

    if len(texts) != info["rows"]:
        raise ValueError(f"{split}: row count mismatch")

    return texts


def tokenize_lengths(tokenizer, texts: list[str]) -> np.ndarray:
    lengths = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch = texts[start : start + BATCH_SIZE]

        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=False,
        )

        lengths.extend(len(input_ids) for input_ids in encoded["input_ids"])

    result = np.asarray(lengths, dtype=np.int32)

    if len(result) != len(texts) or np.any(result <= 0):
        raise ValueError("Invalid token-length output")

    return result


def summarize(lengths: np.ndarray) -> dict:
    summary = {
        "count": int(len(lengths)),
        "min": int(lengths.min()),
        "max": int(lengths.max()),
        "mean": float(lengths.mean()),
        "median": float(np.median(lengths)),
        "percentiles": {
            str(p): float(np.percentile(lengths, p)) for p in (50, 90, 95, 99, 99.5)
        },
        "candidate_lengths": {},
    }

    for limit in CANDIDATE_LENGTHS:
        affected = int(np.count_nonzero(lengths > limit))

        summary["candidate_lengths"][str(limit)] = {
            "examples_truncated": affected,
            "percentage_truncated": float(affected / len(lengths) * 100),
            "tokens_removed": int(np.maximum(lengths - limit, 0).sum()),
        }

    return summary


def main() -> None:
    if REPORT_PATH.exists():
        raise FileExistsError(f"Report already exists: {REPORT_PATH}")

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    if (
        manifest["dataset_version"] != "goemotions_v001"
        or len(manifest["label_names"]) != 28
    ):
        raise ValueError("Unexpected processed dataset")

    revision = HfApi().model_info(MODEL_ID).sha

    if not revision:
        raise ValueError("Unable to resolve tokenizer revision")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=revision,
        use_fast=True,
    )

    if not tokenizer.is_fast:
        raise ValueError("Expected a fast tokenizer")

    report = {
        "analysis_id": "roberta_token_lengths_v001",
        "dataset_version": manifest["dataset_version"],
        "dataset_manifest_sha256": sha256_file(MANIFEST_PATH),
        "tokenizer_model": MODEL_ID,
        "tokenizer_revision": revision,
        "special_tokens_included": True,
        "padding": False,
        "truncation": False,
        "splits": {},
    }

    for split in SPLITS:
        print(f"Analyzing {split}...")

        texts = load_texts(split, manifest)
        lengths = tokenize_lengths(tokenizer, texts)

        summary = summarize(lengths)
        report["splits"][split] = summary

        print(
            f"{split}: {summary['count']} examples, "
            f"mean={summary['mean']:.2f}, "
            f"median={summary['median']:.1f}, "
            f"max={summary['max']}"
        )

        for limit, result in summary["candidate_lengths"].items():
            print(
                f"  max_length={limit:<3} "
                f"truncated={result['examples_truncated']:<5} "
                f"({result['percentage_truncated']:.2f}%)"
            )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nTokenizer revision: {revision}")
    print(f"[OK] Report saved: {REPORT_PATH}")
    print("[OK] P11-D2A token-length analysis completed")


if __name__ == "__main__":
    main()
