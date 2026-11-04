import hashlib
import json
from collections import Counter
from pathlib import Path

from ml.src.data.goemotions_dataset import (
    load_goemotions_rows,
)
from ml.scripts.generate_goemotions_synthetic_v002 import (
    CANDIDATES_PATH,
    MANIFEST_PATH,
    normalize,
)


ROOT = Path(__file__).resolve().parents[2]

AUDIT_PATH = ROOT / "reports/goemotions/synthetic_pilot_v002_audit.json"

SOURCE_MANIFEST = ROOT / "data/processed/goemotions_v001/manifest.json"

OUTPUT_DIR = ROOT / "data/processed/synthetic_goemotions_v002"

DATA_PATH = OUTPUT_DIR / "train.jsonl"
OUTPUT_MANIFEST = OUTPUT_DIR / "manifest.json"

EXCLUDED_IDS = {"syn_v002_nervousness_007_03"}

EXPECTED_COUNTS = {
    "grief": 75,
    "nervousness": 74,
    "pride": 75,
    "relief": 75,
}


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    # Prevent overwriting existing training data.
    if OUTPUT_DIR.exists():
        raise FileExistsError(f"Dataset directory already exists: {OUTPUT_DIR}")

    audit = load_json(AUDIT_PATH)
    source_manifest = load_json(SOURCE_MANIFEST)
    candidates = load_jsonl(CANDIDATES_PATH)
    generation_manifest = load_json(MANIFEST_PATH)

    if audit["candidate_count"] != 300:
        raise ValueError("Unexpected audit candidate count")

    if audit["experiment_id"] != ("goemotions_synthetic_pilot_v002"):
        raise ValueError("Unexpected source experiment")

    if sha256_file(CANDIDATES_PATH) != audit["candidate_file_sha256"]:
        raise ValueError("Candidates changed after audit")

    if sha256_file(MANIFEST_PATH) != audit["generation_manifest_sha256"]:
        raise ValueError("Generation manifest changed")

    if generation_manifest["model"] != "gpt-6.1-terra":
        raise ValueError("Unexpected generation model")

    if source_manifest["dataset_version"] != "goemotions_v001":
        raise ValueError("Unexpected source dataset version")

    names = source_manifest["label_names"]

    if len(names) != 28 or len(set(names)) != 28:
        raise ValueError("Invalid label vocabulary")

    label_to_id = {name: index for index, name in enumerate(names)}

    if len(candidates) != 300:
        raise ValueError("Expected 300 candidates")

    # Validate the known duplicate decision.
    pairs = audit["near_duplicate_pairs"]

    duplicate_verified = any(
        {
            pair["candidate_a"],
            pair["candidate_b"],
        }
        == {
            "syn_v002_nervousness_004_04",
            "syn_v002_nervousness_007_03",
        }
        for pair in pairs
    )

    if not duplicate_verified:
        raise ValueError("Known duplicate not supported by audit")

    # Exclusion-only comparison against real data.
    # Do not use real examples as generation prompts.
    real_texts = set()

    for split in ("train", "validation", "test"):
        real_rows, _ = load_goemotions_rows(split)

        for row in real_rows:
            real_texts.add(normalize(row["text"]))

    approved = []
    seen_ids = set()
    seen_texts = set()
    counts = Counter()

    for candidate in candidates:
        cid = candidate["candidate_id"]

        if cid in seen_ids:
            raise ValueError("Duplicate candidate ID")

        seen_ids.add(cid)

        if cid in EXCLUDED_IDS:
            continue

        text = candidate["text"]
        normalized = normalize(text)

        if not text.strip():
            raise ValueError("Empty synthetic text")

        if normalized in seen_texts or normalized in real_texts:
            raise ValueError(f"Duplicate or real-data overlap: {cid}")

        seen_texts.add(normalized)

        primary = candidate["intended_primary_label"]

        secondary = candidate["proposed_secondary_labels"]

        proposed_labels = [primary] + secondary

        if (
            len(proposed_labels) != len(set(proposed_labels))
            or any(label not in label_to_id for label in proposed_labels)
            or "neutral" in proposed_labels
        ):
            raise ValueError(f"Invalid proposed labels: {cid}")

        if (
            candidate["approval_status"] != "pending"
            or candidate["label_status"] != "unverified"
        ):
            raise ValueError(f"Unexpected source review status: {cid}")

        if (
            candidate["config_sha256"] != generation_manifest["config_sha256"]
            or candidate["prompt_sha256"] != generation_manifest["prompt_sha256"]
        ):
            raise ValueError(f"Candidate provenance mismatch: {cid}")

        counts[primary] += 1

        approved.append(
            {
                "example_id": cid,
                "source_id": cid,
                "text": text,
                "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "labels": sorted(label_to_id[label] for label in proposed_labels),
                "intended_primary_label": primary,
                "label_provenance": "llm_proposed",
                "approval_basis": ("one_time_semantic_review_waiver"),
            }
        )

    if len(approved) != 299:
        raise ValueError("Unexpected approved count")

    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError("Unexpected label counts")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)

    with DATA_PATH.open("x", encoding="utf-8", newline="\n") as stream:
        for row in approved:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    output_manifest = {
        "dataset_version": "synthetic_goemotions_v002",
        "source_experiment": ("goemotions_synthetic_pilot_v002"),
        "source_model": "gpt-6.1-terra",
        "source_manifest_sha256": sha256_file(MANIFEST_PATH),
        "source_candidates_sha256": sha256_file(CANDIDATES_PATH),
        "source_audit_sha256": sha256_file(AUDIT_PATH),
        "real_dataset_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "rows": len(approved),
        "sha256": sha256_file(DATA_PATH),
        "primary_label_counts": dict(counts),
        "excluded_candidate_ids": sorted(EXCLUDED_IDS),
        "review_policy": {
            "type": "one_time_exemption",
            "individual_semantic_review": False,
            "requested_by": "project_supervisor",
            "applies_only_to": ("synthetic_goemotions_v002"),
            "label_source": "generator_proposed",
            "training_use": "experimental",
            "future_datasets_exempt": False,
        },
        "limitations": [
            "Synthetic labels were not independently verified by human annotators.",
            "Automated integrity checks do not prove semantic annotation correctness.",
            "One confirmed near-duplicate excluded.",
            "Generated-text distribution may differ from real GoEmotions comments.",
            "Held-out text was used only for exclusion checks.",
        ],
    }

    with OUTPUT_MANIFEST.open("x", encoding="utf-8") as stream:
        json.dump(
            output_manifest,
            stream,
            indent=2,
            ensure_ascii=False,
        )
        stream.write("\n")

    print("P11-E2F — SYNTHETIC DATASET EXPORT")
    print("-" * 42)
    print("Generated candidates: 300")
    print("Excluded duplicates:", len(EXCLUDED_IDS))
    print("Approved training examples:", len(approved))

    for label, count in counts.items():
        print(f"  {label:<13} {count}")

    print("Semantic review: WAIVED FOR V002 ONLY")
    print("Label provenance: LLM-proposed")
    print("Dataset SHA256:", output_manifest["sha256"])
    print(f"[OK] Dataset saved: {DATA_PATH}")
    print(f"[OK] Manifest saved: {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
