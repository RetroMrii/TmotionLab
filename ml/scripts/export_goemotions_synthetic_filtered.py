import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/synthetic/goemotions_large_scale"
GATE_ROOT = SOURCE / "quality/gate_runs"
LABEL_MANIFEST = ROOT / "data/processed/goemotions_v001/manifest.json"

OUTPUT = ROOT / "data/processed/synthetic_goemotions_opt009_v001"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Output already exists: {OUTPUT}")

    candidates_path = SOURCE / "candidates.jsonl"

    gate_dirs = sorted(
        path
        for path in GATE_ROOT.iterdir()
        if path.is_dir() and (path / "quality_gate_summary.json").exists()
    )

    if not gate_dirs:
        raise RuntimeError("No quality-gate reports found")

    # Use the most recent completed gate snapshot.
    gate = gate_dirs[-1]

    summary_path = gate / "quality_gate_summary.json"
    decisions_path = gate / "quality_decisions.jsonl"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    candidates = read_jsonl(candidates_path)
    decisions = read_jsonl(decisions_path)

    source_hash = sha256_file(candidates_path)

    if summary["candidate_sha256"] != source_hash:
        raise ValueError("Quality-gate report does not match candidate file")

    if len(candidates) != 42000:
        raise ValueError(f"Expected 42000 candidates, got {len(candidates)}")

    candidate_ids = [row["candidate_id"] for row in candidates]
    decision_ids = [row["candidate_id"] for row in decisions]

    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Duplicate candidate IDs")

    if (
        len(decisions) != len(candidates)
        or len(set(decision_ids)) != len(decisions)
        or set(decision_ids) != set(candidate_ids)
    ):
        raise ValueError("Incomplete or mismatched quality decisions")

    label_source = json.loads(LABEL_MANIFEST.read_text(encoding="utf-8"))

    labels = label_source["label_names"]

    if len(labels) != 28 or len(set(labels)) != 28:
        raise ValueError("Invalid GoEmotions label vocabulary")

    label_to_id = {label: index for index, label in enumerate(labels)}

    decision_map = {row["candidate_id"]: row for row in decisions}

    excluded = []
    retained = []
    reasons_count = Counter()
    retained_primary_counts = Counter()

    for candidate in candidates:
        cid = candidate["candidate_id"]
        decision = decision_map[cid]

        reasons = decision.get("reasons", [])

        if not isinstance(reasons, list):
            raise ValueError(f"Invalid review reasons: {cid}")

        if reasons:
            excluded.append(
                {
                    "candidate_id": cid,
                    "status": decision["status"],
                    "reasons": reasons,
                }
            )
            reasons_count.update(reasons)
            continue

        if decision["status"] not in {
            "pending_annotation",
            "provisional_agreement",
        }:
            raise ValueError(
                f"Unexpected unflagged status for {cid}: {decision['status']}"
            )

        primary = candidate["intended_primary_label"]
        secondary = candidate["proposed_secondary_labels"]

        proposed = [primary, *secondary]

        if len(set(proposed)) != len(proposed):
            raise ValueError(f"Repeated labels: {cid}")

        if any(label not in label_to_id for label in proposed):
            raise ValueError(f"Unknown labels: {cid}")

        retained_primary_counts[primary] += 1

        retained.append(
            {
                "example_id": cid,
                "source_id": cid,
                "text": candidate["text"],
                "labels": sorted(label_to_id[label] for label in proposed),
                "intended_primary_label": primary,
                "label_provenance": "llm_proposed",
                "quality_status": decision["status"],
                "annotation_verified": False,
                "source": "synthetic_opt009",
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=False)

    data_path = OUTPUT / "train.jsonl"
    excluded_path = OUTPUT / "excluded.jsonl"
    manifest_path = OUTPUT / "manifest.json"

    write_jsonl(data_path, retained)
    write_jsonl(excluded_path, excluded)

    manifest = {
        "dataset_version": "synthetic_goemotions_opt009_v001",
        "experiment": "OPT-009",
        "source_candidates": len(candidates),
        "retained": len(retained),
        "excluded": len(excluded),
        "source_candidate_sha256": source_hash,
        "quality_gate": str(gate.relative_to(ROOT)),
        "quality_gate_summary_sha256": sha256_file(summary_path),
        "quality_decisions_sha256": sha256_file(decisions_path),
        "training_data_sha256": sha256_file(data_path),
        "excluded_data_sha256": sha256_file(excluded_path),
        "label_manifest_sha256": sha256_file(LABEL_MANIFEST),
        "retained_primary_counts": dict(retained_primary_counts),
        "exclusion_reasons": dict(reasons_count),
        "filter_policy": (
            "Exclude all candidates with any recorded "
            "integrity, similarity or annotation review reason."
        ),
        "annotation_provenance": "generator_proposed",
        "human_review": (
            "Calibration sample only; retained examples "
            "are not individually human verified."
        ),
        "training_approval": False,
        "limitations": [
            "Synthetic labels are not independently verified for every row.",
            "No automatic semantic correction of proposed labels.",
            "Embedding similarity screening is approximate.",
            "Unflagged candidates may still contain annotation errors.",
            "Validation/test data used only for exclusion screening.",
        ],
    }

    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")

    print("\nOPT-009 — FILTERED DATASET EXPORT")
    print("=" * 48)
    print("Original candidates:", len(candidates))
    print("Excluded candidates:", len(excluded))
    print("Retained candidates:", len(retained))
    print("Exclusion reasons:", dict(reasons_count))
    print("Dataset:", data_path)
    print("Exclusions:", excluded_path)
    print("Manifest:", manifest_path)
    print("Training approval: NOT GRANTED")


if __name__ == "__main__":
    main()
