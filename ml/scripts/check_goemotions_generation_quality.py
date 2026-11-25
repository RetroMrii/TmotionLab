import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data/synthetic/goemotions_large_scale"

CANDIDATES = DATA_DIR / "candidates.jsonl"
BATCH_DIR = DATA_DIR / "batches"
QUALITY_DIR = DATA_DIR / "quality"
GATE_DIR = QUALITY_DIR / "gate_runs"

INTEGRITY_REPORT = QUALITY_DIR / "integrity_audit.json"
DECISIONS = QUALITY_DIR / "quality_decisions.jsonl"

SEMANTIC_MODULE = "ml.scripts.audit_goemotions_semantic_duplicates"

QUALITY_MODULE = "ml.scripts.quality_goemotions_large_scale"


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(block)

    return digest.hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run_command(command):
    print("\nRunning:", " ".join(map(str, command)), flush=True)

    result = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(f"Audit command exited with code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--synthetic-threshold",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--real-threshold",
        type=float,
        default=0.88,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    args = parser.parse_args()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

    run_dir = GATE_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    summary_path = run_dir / "quality_gate_summary.json"

    reasons = []
    statistics = {}
    source_hash = None

    try:
        if not CANDIDATES.is_file():
            raise FileNotFoundError(CANDIDATES)

        source_hash = sha256_file(CANDIDATES)
        candidates = load_jsonl(CANDIDATES)

        candidate_ids = [row["candidate_id"] for row in candidates]

        if not candidates:
            reasons.append("No candidates available")

        if len(candidate_ids) != len(set(candidate_ids)):
            reasons.append("Duplicate candidate IDs")

        statistics["candidate_count"] = len(candidates)

        # Verify candidates were rebuilt from all completed
        # generation batches, rather than a stale subset.
        batch_paths = sorted(BATCH_DIR.glob("*.json"))
        expected_rows = 0

        for path in batch_paths:
            batch = load_json(path)
            expected_rows += len(batch["parsed"]["examples"])

        statistics["completed_batches"] = len(batch_paths)
        statistics["expected_candidate_count"] = expected_rows

        if expected_rows != len(candidates):
            reasons.append(
                "Candidate count does not match completed "
                f"batches: expected {expected_rows}, "
                f"found {len(candidates)}"
            )

        if not batch_paths:
            reasons.append("No completed generation batches")

        # Do not run expensive semantic search if the input
        # is already incomplete or structurally inconsistent.
        if reasons:
            raise RuntimeError("Preflight failed: " + "; ".join(reasons))

        semantic_dir = run_dir / "semantic"

        run_command(
            [
                sys.executable,
                "-m",
                SEMANTIC_MODULE,
                "--input",
                str(CANDIDATES),
                "--output-dir",
                str(semantic_dir),
                "--synthetic-threshold",
                str(args.synthetic_threshold),
                "--real-threshold",
                str(args.real_threshold),
                "--top-k",
                str(args.top_k),
                "--batch-size",
                str(args.batch_size),
            ]
        )

        run_command(
            [
                sys.executable,
                "-m",
                QUALITY_MODULE,
                "--audit",
                "--input",
                str(CANDIDATES),
                "--semantic-dir",
                str(semantic_dir),
            ]
        )

        semantic = load_json(semantic_dir / "report.json")
        integrity = load_json(INTEGRITY_REPORT)
        decisions = load_jsonl(DECISIONS)
        semantic_flags = load_jsonl(semantic_dir / "candidate_flags.jsonl")

        # Verify that the audit belongs to this exact
        # generation snapshot.
        if sha256_file(CANDIDATES) != source_hash:
            reasons.append("Candidate file changed while audits were running")

        if semantic["candidate_file_sha256"] != source_hash:
            reasons.append("Semantic report has a different candidate hash")

        if semantic["candidate_count"] != len(candidates):
            reasons.append("Semantic audit processed an incorrect count")

        if integrity["candidate_count"] != len(candidates):
            reasons.append("Integrity audit processed an incorrect count")

        decision_ids = [row["candidate_id"] for row in decisions]

        flag_ids = [row["candidate_id"] for row in semantic_flags]

        if (
            len(decisions) != len(candidates)
            or len(set(decision_ids)) != len(decisions)
            or set(decision_ids) != set(candidate_ids)
        ):
            reasons.append("Quality decisions do not match candidate IDs")

        if (
            len(semantic_flags) != len(candidates)
            or len(set(flag_ids)) != len(semantic_flags)
            or set(flag_ids) != set(candidate_ids)
        ):
            reasons.append("Semantic flags do not match candidate IDs")

        # The quality gate is deliberately conservative:
        # unresolved integrity and semantic flags fail.
        integrity_flags = integrity.get("flag_counts", {})

        for flag, count in sorted(integrity_flags.items()):
            if count:
                reasons.append(f"Integrity: {flag} ({count} candidates)")

        synthetic_pairs = semantic["synthetic_similarity_pairs"]
        real_matches = semantic["real_similarity_matches"]

        if synthetic_pairs:
            reasons.append(
                f"Semantic: {synthetic_pairs} suspicious synthetic-to-synthetic pairs"
            )

        if real_matches:
            reasons.append(
                f"Semantic: {real_matches} suspicious synthetic-to-GoEmotions matches"
            )

        status_counts = Counter(row["status"] for row in decisions)

        statistics.update(
            {
                "integrity_flag_counts": integrity_flags,
                "synthetic_similarity_pairs": synthetic_pairs,
                "real_similarity_matches": real_matches,
                "semantic_flagged_candidates": semantic["flagged_candidate_count"],
                "quality_status_counts": dict(status_counts),
                "candidate_file_sha256": source_hash,
                "semantic_thresholds": {
                    "synthetic": args.synthetic_threshold,
                    "real": args.real_threshold,
                },
            }
        )

        for status in ("rejected_integrity", "needs_review"):
            count = status_counts.get(status, 0)

            if count:
                reasons.append(f"Quality decisions: {count} {status}")

        # Preserve reports from this run.
        shutil.copy2(
            INTEGRITY_REPORT,
            run_dir / "integrity_audit.json",
        )
        shutil.copy2(
            DECISIONS,
            run_dir / "quality_decisions.jsonl",
        )

    except Exception as error:
        reasons.append(f"Audit execution: {type(error).__name__}: {error}")

    passed = len(reasons) == 0

    summary = {
        "experiment": "OPT-009",
        "stage": "generation_quality_gate",
        "result": "PASS" if passed else "FAIL",
        "scope": [
            "structural_integrity",
            "normalized_exact_overlap",
            "embedding_similarity_screening",
        ],
        "candidate_sha256": source_hash,
        "statistics": statistics,
        "reasons": reasons,
        "limitations": [
            "PASS does not verify emotion-label correctness.",
            "PASS does not approve candidates for training.",
            "Embedding similarity search is approximate.",
            "Similarity thresholds are provisional.",
            "Novel paraphrases may evade detection.",
            "Human quality sampling remains required.",
        ],
    }

    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 52)
    print("OPT-009 — GENERATION QUALITY GATE")
    print("=" * 52)

    print("RESULT:", summary["result"])
    print("Candidates:", statistics.get("candidate_count", 0))
    print(
        "Completed batches:",
        statistics.get("completed_batches", 0),
    )

    if reasons:
        print("\nREASONS:")
        for number, reason in enumerate(reasons, start=1):
            print(f"{number}. {reason}")
    else:
        print(
            "\nNo integrity or semantic similarity "
            "flags detected at configured thresholds."
        )

    print("\nReport:", summary_path)
    print("Training approval: NOT GRANTED")

    # Suitable for automated pipelines.
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
