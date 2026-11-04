import argparse
import hashlib
import json
import os
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from ml.scripts.generate_goemotions_synthetic_pilot import (
    normalize,
)


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "ml/configs/synthetic_goemotions_pilot_v002.json"
PROMPT_PATH = ROOT / "ml/configs/synthetic/synthetic_emotion_generation_v002.txt"
OUTPUT_DIR = ROOT / "data/synthetic/goemotions_pilot_v002"
BATCH_DIR = OUTPUT_DIR / "batches"
MANIFEST_PATH = OUTPUT_DIR / "generation_manifest.json"
CANDIDATES_PATH = OUTPUT_DIR / "candidates.jsonl"

LABEL_NAMES = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)


class GeneratedExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario_slot: int
    text: str
    secondary_labels: list[str]


class GeneratedBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    examples: list[GeneratedExample]


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def load_spec():
    config = load_json(CONFIG_PATH)
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    if config["experiment_id"] != "goemotions_synthetic_pilot_v002":
        raise ValueError("Unexpected experiment ID")

    if config["model"] != "gpt-6.1-terra":
        raise ValueError("Unexpected generator model")

    if config["examples_per_request"] != 5:
        raise ValueError("Expected batches of five")

    expected_labels = {"grief", "nervousness", "pride", "relief"}

    if (
        set(config["targets"]) != expected_labels
        or set(config["scenario_pools"]) != expected_labels
    ):
        raise ValueError("Unexpected target labels")

    for label, count in config["targets"].items():
        if count != 75:
            raise ValueError(f"Unexpected target count: {label}")

        pool = config["scenario_pools"][label]

        if len(pool) != 15 or len(set(pool)) != 15:
            raise ValueError(f"Invalid scenario pool: {label}")

    return config, prompt


def build_jobs(config):
    jobs = []

    for label, count in config["targets"].items():
        pool = config["scenario_pools"][label]

        for batch_index in range(count // config["examples_per_request"]):
            scenarios = [
                pool[(batch_index + slot * 3) % len(pool)] for slot in range(5)
            ]

            jobs.append(
                {
                    "batch_id": f"{label}_{batch_index:03d}",
                    "label": label,
                    "batch_index": batch_index,
                    "scenarios": scenarios,
                }
            )

    return jobs


def build_request(base_prompt, job):
    assignments = "\n".join(
        f"Slot {slot}: {theme}" for slot, theme in enumerate(job["scenarios"])
    )

    return (
        base_prompt
        + "\n\nPRIMARY EMOTION: "
        + job["label"]
        + "\nBATCH ID: "
        + job["batch_id"]
        + "\nASSIGNED SCENARIOS:\n"
        + assignments
        + "\n\nALLOWED SECONDARY LABELS:\n"
        + ", ".join(name for name in LABEL_NAMES if name != "neutral")
        + "\n\nReturn five examples with scenario_slot "
        "values 0, 1, 2, 3, 4, in that order."
    )


def validate_batch(parsed, job):
    if len(parsed.examples) != 5:
        raise ValueError("Expected five examples")

    texts = set()
    allowed = set(LABEL_NAMES) - {job["label"], "neutral"}

    for slot, example in enumerate(parsed.examples):
        if example.scenario_slot != slot:
            raise ValueError("Scenario slot mismatch")

        text = example.text.strip()
        normalized = normalize(text)

        if len(text) < 15 or len(text) > 1200 or len(text.split()) > 150:
            raise ValueError("Invalid text length")

        if normalized in texts:
            raise ValueError("Duplicate text within batch")

        texts.add(normalized)

        secondary = example.secondary_labels

        if len(secondary) != len(set(secondary)) or not set(secondary).issubset(
            allowed
        ):
            raise ValueError("Invalid secondary labels")


def manifest_for(config, jobs):
    return {
        "experiment_id": config["experiment_id"],
        "model": config["model"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "prompt_sha256": sha256_file(PROMPT_PATH),
        "planned_batches": len(jobs),
        "planned_candidates": 300,
        "status": "UNAPPROVED",
    }


def verify_existing(record, job, manifest):
    if (
        record["manifest"] != manifest
        or record["batch_id"] != job["batch_id"]
        or record["primary_label"] != job["label"]
        or record["requested_model"] != manifest["model"]
        or record["scenarios"] != job["scenarios"]
    ):
        raise ValueError("Saved batch provenance mismatch")

    parsed = GeneratedBatch.model_validate(record["parsed_response"])
    validate_batch(parsed, job)

    return parsed


def reconstruct_candidates(jobs, manifest):
    records = []
    candidates = []

    for job in jobs:
        path = BATCH_DIR / f"{job['batch_id']}.json"

        if not path.exists():
            continue

        record = load_json(path)
        parsed = verify_existing(record, job, manifest)
        records.append(record)

        for slot, example in enumerate(parsed.examples):
            candidates.append(
                {
                    "candidate_id": (f"syn_v002_{job['batch_id']}_{slot:02d}"),
                    "text": example.text.strip(),
                    "intended_primary_label": job["label"],
                    "proposed_secondary_labels": (example.secondary_labels),
                    "assigned_scenario": job["scenarios"][slot],
                    "approval_status": "pending",
                    "label_status": "unverified",
                    "generator_model": record["returned_model"],
                    "response_id": record["response_id"],
                    "batch_id": job["batch_id"],
                    "scenario_slot": slot,
                    "prompt_sha256": manifest["prompt_sha256"],
                    "config_sha256": manifest["config_sha256"],
                }
            )

    # Derived file; safe to regenerate from saved batch records.
    temporary = OUTPUT_DIR / "candidates.jsonl.tmp"

    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for candidate in candidates:
            stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    os.replace(temporary, CANDIDATES_PATH)

    return records, candidates


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)

    modes.add_argument("--plan", action="store_true")
    modes.add_argument("--max-batches", type=int)
    modes.add_argument("--all", action="store_true")

    args = parser.parse_args()

    config, prompt = load_spec()
    jobs = build_jobs(config)
    manifest = manifest_for(config, jobs)

    if args.plan:
        print(json.dumps(manifest, indent=2))
        print("Model:", config["model"])
        print("Batches:", len(jobs))
        print("Candidates:", len(jobs) * 5)
        return

    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing")

    if MANIFEST_PATH.exists():
        if load_json(MANIFEST_PATH) != manifest:
            raise ValueError("Generation specification has changed")
    else:
        if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
            raise RuntimeError("Existing outputs without manifest")

        save_json_once(MANIFEST_PATH, manifest)

    BATCH_DIR.mkdir(parents=True, exist_ok=True)

    records, _ = reconstruct_candidates(jobs, manifest)
    completed = {r["batch_id"] for r in records}

    remaining = [job for job in jobs if job["batch_id"] not in completed]

    limit = len(remaining) if args.all else min(args.max_batches, len(remaining))

    client = OpenAI(max_retries=2, timeout=120)

    for job in remaining[:limit]:
        request_prompt = build_request(prompt, job)

        response = client.responses.parse(
            model=config["model"],
            input=[
                {
                    "role": "user",
                    "content": request_prompt,
                }
            ],
            text_format=GeneratedBatch,
            store=False,
        )

        parsed = response.output_parsed

        if parsed is None:
            raise RuntimeError("Missing parsed API response")

        validate_batch(parsed, job)

        record = {
            "manifest": manifest,
            "batch_id": job["batch_id"],
            "primary_label": job["label"],
            "scenarios": job["scenarios"],
            "requested_model": config["model"],
            "returned_model": response.model,
            "response_id": response.id,
            "request_prompt": request_prompt,
            "parsed_response": parsed.model_dump(mode="json"),
            "raw_response": response.model_dump(mode="json"),
            "usage": (
                response.usage.model_dump(mode="json") if response.usage else None
            ),
        }

        path = BATCH_DIR / f"{job['batch_id']}.json"
        save_json_once(path, record)

        completed.add(job["batch_id"])
        print(f"[OK] {job['batch_id']} ({len(completed)}/{len(jobs)})")

        records, candidates = reconstruct_candidates(jobs, manifest)

    records, candidates = reconstruct_candidates(jobs, manifest)

    total_input = sum((r["usage"] or {}).get("input_tokens", 0) for r in records)
    total_output = sum((r["usage"] or {}).get("output_tokens", 0) for r in records)

    estimated_cost = (total_input * 1.00 + total_output * 5.00) / 1_000_000

    print(f"Candidates: {len(candidates)}/300")
    print(f"Input tokens: {total_input}")
    print(f"Output tokens: {total_output}")
    print(f"Estimated standard API cost: ${estimated_cost:.4f}")
    print("Status: UNAPPROVED — not training data")


if __name__ == "__main__":
    main()
