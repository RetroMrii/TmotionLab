import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from ml.src.data.goemotions_dataset import load_goemotions_rows


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT = ROOT / "data/synthetic/goemotions_large_scale/candidates.jsonl"

DEFAULT_OUTPUT = ROOT / "data/synthetic/goemotions_large_scale/quality/semantic"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SYNTHETIC_THRESHOLD = 0.90
REAL_THRESHOLD = 0.88

TOP_K = 12
BATCH_SIZE = 128


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def read_jsonl(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []

    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at line {number}") from error

            rows.append(row)

    return rows


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            data,
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_candidates(rows):
    ids = set()

    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Candidate {index} is not an object")

        cid = row.get("candidate_id")
        text = row.get("text")

        if not isinstance(cid, str) or not cid:
            raise ValueError(f"Invalid candidate ID at {index}")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Invalid candidate text: {cid}")

        if cid in ids:
            raise ValueError(f"Duplicate candidate ID: {cid}")

        ids.add(cid)


def load_real_data():
    texts = []
    metadata = []

    for split in ("train", "validation", "test"):
        rows, _ = load_goemotions_rows(split)

        for index, row in enumerate(rows):
            text = row["text"]

            if not isinstance(text, str):
                raise ValueError(f"Invalid real text: {split}/{index}")

            texts.append(text)

            # Do not store held-out text in the report.
            metadata.append(
                {
                    "split": split,
                    "row_index": index,
                }
            )

    return texts, metadata


def encode_texts(model, texts, batch_size):
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    vectors = np.ascontiguousarray(
        vectors,
        dtype=np.float32,
    )

    if vectors.ndim != 2:
        raise ValueError("Invalid embedding shape")

    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite embeddings")

    lengths = np.linalg.norm(vectors, axis=1)

    if np.any(lengths < 0.99):
        raise ValueError("Unexpected embedding norms")

    return vectors


def make_index(vectors, ef_search):
    import faiss

    dimensions = vectors.shape[1]

    # Approximate nearest-neighbor search with
    # cosine similarity via normalized inner product.
    index = faiss.IndexHNSWFlat(
        dimensions,
        32,
        faiss.METRIC_INNER_PRODUCT,
    )

    index.hnsw.efConstruction = 100
    index.hnsw.efSearch = ef_search

    index.add(vectors)

    return index


def search_batched(index, vectors, top_k, batch_size):
    for start in range(0, len(vectors), batch_size):
        end = min(start + batch_size, len(vectors))

        similarities, neighbors = index.search(
            vectors[start:end],
            top_k,
        )

        yield start, similarities, neighbors


def compare_synthetic(
    vectors,
    candidates,
    threshold,
    top_k,
    batch_size,
    ef_search,
):
    if len(vectors) < 2:
        return []

    index = make_index(vectors, ef_search)

    pairs = {}
    requested_k = min(top_k + 1, len(vectors))

    for start, scores, neighbors in search_batched(
        index,
        vectors,
        requested_k,
        batch_size,
    ):
        for local_index, (values, indices) in enumerate(zip(scores, neighbors)):
            source = start + local_index

            for score, target in zip(values, indices):
                target = int(target)

                if target < 0 or source == target:
                    continue

                score = float(score)

                if score < threshold:
                    continue

                left, right = sorted((source, target))

                key = (left, right)

                if key not in pairs or score > pairs[key]:
                    pairs[key] = score

    result = []

    for (left, right), score in pairs.items():
        result.append(
            {
                "candidate_a": candidates[left]["candidate_id"],
                "candidate_b": candidates[right]["candidate_id"],
                "cosine_similarity": round(score, 6),
                "review_reason": "synthetic_semantic_similarity",
            }
        )

    result.sort(key=lambda row: -row["cosine_similarity"])

    return result


def compare_real(
    real_vectors,
    synthetic_vectors,
    candidates,
    real_metadata,
    threshold,
    top_k,
    batch_size,
    ef_search,
):
    index = make_index(real_vectors, ef_search)

    matches = []

    for start, scores, neighbors in search_batched(
        index,
        synthetic_vectors,
        min(top_k, len(real_vectors)),
        batch_size,
    ):
        for local_index, (values, indices) in enumerate(zip(scores, neighbors)):
            candidate = candidates[start + local_index]

            for score, target in zip(values, indices):
                target = int(target)

                if target < 0:
                    continue

                score = float(score)

                if score < threshold:
                    continue

                source = real_metadata[target]

                matches.append(
                    {
                        "candidate_id": candidate["candidate_id"],
                        "real_split": source["split"],
                        "real_row_index": source["row_index"],
                        "cosine_similarity": round(score, 6),
                        "review_reason": "real_data_semantic_similarity",
                    }
                )

    matches.sort(key=lambda row: -row["cosine_similarity"])

    return matches


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--model",
        default=MODEL_NAME,
    )

    parser.add_argument(
        "--synthetic-threshold",
        type=float,
        default=SYNTHETIC_THRESHOLD,
    )

    parser.add_argument(
        "--real-threshold",
        type=float,
        default=REAL_THRESHOLD,
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=TOP_K,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
    )

    parser.add_argument(
        "--ef-search",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    if not 0 < args.synthetic_threshold <= 1:
        raise ValueError("Invalid synthetic threshold")

    if not 0 < args.real_threshold <= 1:
        raise ValueError("Invalid real threshold")

    if args.top_k < 1 or args.batch_size < 1:
        raise ValueError("Invalid search settings")

    if args.ef_search < 1:
        raise ValueError("Invalid ef-search")

    if args.output_dir.exists():
        raise FileExistsError(
            "Output directory exists. Use a new "
            "directory to preserve previous reports: "
            f"{args.output_dir}"
        )

    candidates = read_jsonl(args.input)
    validate_candidates(candidates)

    if not candidates:
        raise ValueError("No candidates to audit")

    print("OPT-009 — SEMANTIC SIMILARITY AUDIT")
    print("Synthetic candidates:", len(candidates))

    real_texts, real_metadata = load_real_data()

    print("Real reference comments:", len(real_texts))

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)

    print("\nEncoding synthetic comments")

    synthetic_vectors = encode_texts(
        model,
        [row["text"] for row in candidates],
        args.batch_size,
    )

    print("\nEncoding real GoEmotions comments")

    real_vectors = encode_texts(
        model,
        real_texts,
        args.batch_size,
    )

    print("\nSearching synthetic-to-synthetic matches")

    synthetic_pairs = compare_synthetic(
        synthetic_vectors,
        candidates,
        args.synthetic_threshold,
        args.top_k,
        args.batch_size,
        args.ef_search,
    )

    print("\nSearching synthetic-to-real matches")

    real_matches = compare_real(
        real_vectors,
        synthetic_vectors,
        candidates,
        real_metadata,
        args.real_threshold,
        args.top_k,
        args.batch_size,
        args.ef_search,
    )

    flags = {}

    for row in candidates:
        flags[row["candidate_id"]] = set()

    for pair in synthetic_pairs:
        flags[pair["candidate_a"]].add("synthetic_semantic_similarity")
        flags[pair["candidate_b"]].add("synthetic_semantic_similarity")

    for match in real_matches:
        flags[match["candidate_id"]].add("real_data_semantic_similarity")

    flag_rows = [
        {
            "candidate_id": cid,
            "flags": sorted(values),
            "requires_review": bool(values),
            "approved_for_training": False,
        }
        for cid, values in flags.items()
    ]

    flagged_counts = Counter()

    for row in flag_rows:
        flagged_counts.update(row["flags"])

    report = {
        "experiment": "OPT-009",
        "stage": "semantic_similarity_audit",
        "candidate_count": len(candidates),
        "reference_count": len(real_texts),
        "candidate_file_sha256": sha256_file(args.input),
        "embedding_model": args.model,
        "embedding_dimensions": int(synthetic_vectors.shape[1]),
        "search_method": "FAISS HNSW approximate",
        "synthetic_threshold": args.synthetic_threshold,
        "real_threshold": args.real_threshold,
        "top_k": args.top_k,
        "ef_search": args.ef_search,
        "synthetic_similarity_pairs": len(synthetic_pairs),
        "real_similarity_matches": len(real_matches),
        "flagged_candidate_count": sum(row["requires_review"] for row in flag_rows),
        "flag_counts": dict(flagged_counts),
        "limitations": [
            "Embedding similarity is not semantic identity.",
            "HNSW is approximate and may miss neighbors.",
            "Thresholds are provisional and not calibrated.",
            "Only the top-k retrieved neighbors are inspected.",
            "No candidates are automatically approved or rejected.",
            "Protected real data is used solely for exclusion screening.",
            "This is not an independent emotion-label audit.",
        ],
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)

    write_json(
        output / "report.json",
        report,
    )

    write_jsonl(
        output / "synthetic_pairs.jsonl",
        synthetic_pairs,
    )

    write_jsonl(
        output / "real_matches.jsonl",
        real_matches,
    )

    write_jsonl(
        output / "candidate_flags.jsonl",
        flag_rows,
    )

    print("\nRESULTS")
    print("Synthetic similarity pairs:", len(synthetic_pairs))
    print("Real-data similarity matches:", len(real_matches))
    print(
        "Flagged candidates:",
        report["flagged_candidate_count"],
    )
    print("Saved:", output)
    print("No candidates approved for training.")


if __name__ == "__main__":
    main()
