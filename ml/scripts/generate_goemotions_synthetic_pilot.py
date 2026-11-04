import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, ConfigDict


ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = ROOT / "ml/configs/synthetic_goemotions_pilot_v001.json"
PROMPT_PATH = ROOT / "ml/configs/synthetic/synthetic_emotion_generation_v001.txt"
OUTPUT_DIR = ROOT / "data/synthetic/goemotions_pilot_v001"

META_PATH = OUTPUT_DIR / "generation_manifest.json"
JOURNAL_PATH = OUTPUT_DIR / "batches.jsonl"
CANDIDATES_PATH = OUTPUT_DIR / "candidates.jsonl"

STYLE_PROFILES = (
    "Brief, understated, conversational.",
    "Indirect emotion, everyday concrete situations.",
    "Varied sentence structure and moderate length.",
    "Informal first-person reflection.",
    "Specific situational context, without clichés.",
)


class GeneratedExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    secondary_labels: list[str]


class GeneratedBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    examples: list[GeneratedExample]


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    return sha256_bytes(path.read_bytes())


def canonical_json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def normalize(text):
    return " ".join(text.casefold().split())


def load_settings():
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    prompt = PROMPT_PATH.read_text(encoding="utf-8")

    if config["experiment_id"] != "goemotions_synthetic_pilot_v001":
        raise ValueError("Unexpected experiment ID")

    if config["prompt_version"] != PROMPT_PATH.stem:
        raise ValueError("Prompt version mismatch")

    size = config["examples_per_request"]
    targets = config["targets"]

    if size != 5 or set(targets) != {"grief", "nervousness", "pride", "relief"}:
        raise ValueError("Unexpected pilot settings")

    if any(count <= 0 or count % size for count in targets.values()):
        raise ValueError("Invalid target batch sizes")

    return config, prompt


def build_plan(config):
    jobs = []
    size = config["examples_per_request"]

    for label, count in config["targets"].items():
        for ordinal in range(count // size):
            jobs.append(
                {
                    "batch_id": f"{label}_{ordinal:03d}",
                    "label": label,
                    "ordinal": ordinal,
                    "count": size,
                }
            )

    return jobs


def read_jsonl(path):
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_examples(parsed, job, config):
    examples = parsed.examples

    if len(examples) != job["count"]:
        raise ValueError("Incorrect number of examples")

    allowed = set(config["allowed_secondary_labels"])
    seen = set()

    for example in examples:
        text = example.text.strip()
        normalized = normalize(text)

        if not text or len(text) < 15 or len(text) > 1200 or len(text.split()) > 150:
            raise ValueError("Invalid comment length")

        if normalized in seen:
            raise ValueError("Duplicate within batch")

        seen.add(normalized)

        secondary = example.secondary_labels

        if (
            len(set(secondary)) != len(secondary)
            or not set(secondary).issubset(allowed)
            or job["label"] in secondary
            or ("neutral" in secondary and len(secondary) > 1)
        ):
            raise ValueError("Invalid secondary labels")


def build_prompt(prompt, config, job):
    profile_index = (job["ordinal"] + config["seed"]) % len(STYLE_PROFILES)

    return (
        f"{prompt}\n\n"
        f"PRIMARY EMOTION: {job['label']}\n"
        f"NUMBER OF COMMENTS: {job['count']}\n"
        f"SEMANTIC CONTRAST LABELS: "
        f"{', '.join(config['contrast_labels'][job['label']])}\n"
        f"SECONDARY LABEL VOCABULARY: "
        f"{', '.join(config['allowed_secondary_labels'])}\n"
        f"WRITING STYLE: {STYLE_PROFILES[profile_index]}\n"
        f"BATCH ID: {job['batch_id']}\n"
        "Make the five situations distinct from one another.\n"
    )


def expected_manifest(config, jobs):
    return {
        "experiment_id": config["experiment_id"],
        "model": config["model"],
        "config_sha256": sha256_file(CONFIG_PATH),
        "prompt_sha256": sha256_file(PROMPT_PATH),
        "jobs_sha256": sha256_bytes(canonical_json(jobs).encode("utf-8")),
        "planned_batches": len(jobs),
        "planned_candidates": sum(job["count"] for job in jobs),
        "dataset_status": "UNAPPROVED_CANDIDATES",
        "generation_only": True,
    }


def load_or_initialize_manifest(expected):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if META_PATH.exists():
        actual = json.loads(META_PATH.read_text(encoding="utf-8"))

        if actual != expected:
            raise ValueError(
                "Generation specification changed. Use a new experiment version."
            )
    else:
        if JOURNAL_PATH.exists() or CANDIDATES_PATH.exists():
            raise RuntimeError("Artifacts exist without manifest.")

        with META_PATH.open("x", encoding="utf-8") as stream:
            json.dump(expected, stream, indent=2)
            stream.write("\n")


def validated_journal(jobs, config, expected):
    records = read_jsonl(JOURNAL_PATH)

    jobs_by_id = {job["batch_id"]: job for job in jobs}

    completed = set()

    for record in records:
        batch_id = record["batch_id"]

        if batch_id in completed:
            raise ValueError("Duplicate completed batch")

        if batch_id not in jobs_by_id:
            raise ValueError("Unexpected batch ID")

        # Validate provenance inside the loop.
        if (
            record["primary_label"] != jobs_by_id[batch_id]["label"]
            or record["model"] != config["model"]
        ):
            raise ValueError("Batch label or model mismatch")

        if (
            record["config_sha256"] != expected["config_sha256"]
            or record["prompt_sha256"] != expected["prompt_sha256"]
        ):
            raise ValueError("Batch provenance mismatch")

        parsed = GeneratedBatch.model_validate(record["parsed"])

        validate_examples(
            parsed,
            jobs_by_id[batch_id],
            config,
        )

        completed.add(batch_id)

    return records, completed


def rebuild_candidates(records, jobs):
    job_order = {job["batch_id"]: index for index, job in enumerate(jobs)}

    records = sorted(records, key=lambda record: job_order[record["batch_id"]])

    temporary = CANDIDATES_PATH.with_suffix(".jsonl.tmp")

    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            label = record["primary_label"]

            for position, item in enumerate(record["parsed"]["examples"]):
                candidate = {
                    "candidate_id": (f"syn_{record['batch_id']}_{position:02d}"),
                    "text": item["text"].strip(),
                    "intended_primary_label": label,
                    "proposed_secondary_labels": (item["secondary_labels"]),
                    "label_status": "unverified",
                    "approval_status": "pending",
                    "batch_id": record["batch_id"],
                    "response_id": record["response_id"],
                    "generator_model": record["model"],
                    "prompt_sha256": record["prompt_sha256"],
                    "config_sha256": record["config_sha256"],
                }

                stream.write(json.dumps(candidate, ensure_ascii=False) + "\n")

    temporary.replace(CANDIDATES_PATH)


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--max-batches", type=int)
    mode.add_argument("--all", action="store_true")
    args = parser.parse_args()

    config, prompt = load_settings()
    jobs = build_plan(config)
    expected = expected_manifest(config, jobs)

    if args.plan:
        print(
            f"Planned batches: {len(jobs)}\n"
            f"Planned candidates: "
            f"{expected['planned_candidates']}\n"
            f"Model: {config['model']}"
        )
        return

    if args.max_batches is not None and args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    load_or_initialize_manifest(expected)
    records, completed = validated_journal(jobs, config, expected)

    # Repair the derived file after any interrupted run.
    rebuild_candidates(records, jobs)

    remaining = [job for job in jobs if job["batch_id"] not in completed]

    limit = len(remaining) if args.all else min(args.max_batches, len(remaining))

    client = OpenAI(max_retries=2, timeout=120)

    for job in remaining[:limit]:
        request_prompt = build_prompt(prompt, config, job)
        prompt_hash = sha256_bytes(request_prompt.encode("utf-8"))

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
            raise RuntimeError(f"No parsed response: {job['batch_id']}")

        validate_examples(parsed, job, config)

        record = {
            "batch_id": job["batch_id"],
            "primary_label": job["label"],
            "model": response.model,
            "response_id": response.id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_sha256": expected["config_sha256"],
            "prompt_sha256": expected["prompt_sha256"],
            "request_prompt": request_prompt,
            "request_prompt_sha256": prompt_hash,
            "parsed": parsed.model_dump(mode="json"),
            "raw_response": response.model_dump(mode="json"),
        }

        append_jsonl(JOURNAL_PATH, record)
        records.append(record)
        completed.add(job["batch_id"])

        rebuild_candidates(records, jobs)

        print(f"[OK] {job['batch_id']} ({len(completed)}/{len(jobs)} batches)")

    print(
        f"[OK] {len(completed) * config['examples_per_request']}"
        f"/{expected['planned_candidates']} "
        "candidates journaled"
    )
    print(f"Candidates: {CANDIDATES_PATH}")
    print("Status: UNAPPROVED — not training data")


if __name__ == "__main__":
    main()
