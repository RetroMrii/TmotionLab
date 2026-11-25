"""Generate unapproved synthetic GoEmotions candidates with explicit budget control.

Install: pip install openai pydantic
From C:\\TmotionLab: python -m ml.scripts.generate_goemotions_large_scale --plan
No API calls without --run plus spending arguments.
"""

import argparse
import hashlib
import json
import os
import random
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ml/configs/synthetic_goemotions_large_scale.json"
PROMPT = ROOT / "ml/configs/synthetic/synthetic_emotion_generation_large_scale.txt"

LENGTH_LIMITS = {
    "brief": (4, 18),
    "short": (16, 35),
    "medium": (36, 70),
    "long": (71, 110),
}
DIFFICULTY_MAP = {"explicit": "easy", "implicit": "moderate", "subtle_contrast": "hard"}
WORD_PATTERN = re.compile(r"\b[\w]+(?:['’][\w]+)*\b", flags=re.UNICODE)


class Example(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario_slot: int
    text: str
    secondary_labels: list[str]


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    examples: list[Example]


def canonical(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load_spec():
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    prompt = PROMPT.read_text(encoding="utf-8")
    quotas = cfg["primary_label_quotas"]
    allowed = cfg["allowed_secondary_labels"]
    size = cfg["examples_per_request"]
    if len(allowed) != 28 or len(set(allowed)) != 28 or "neutral" not in allowed:
        raise ValueError("Expected 28 unique canonical labels")
    if len(quotas) != 28 or set(quotas) != set(allowed):
        raise ValueError("Primary label quotas must cover all 28 labels")
    if sum(quotas.values()) != cfg["candidate_target"]:
        raise ValueError("Quota total mismatch")
    if not 2 <= size <= 20 or any(x <= 0 or x % size for x in quotas.values()):
        raise ValueError("Invalid per-request size or quota divisibility")
    if (
        cfg["dataset_status"] != "UNAPPROVED_CANDIDATES"
        or cfg["generation_only"] is not True
    ):
        raise ValueError("Generation-only safety contract violated")
    if cfg["quality_policy"]["autoapprove"] is not False:
        raise ValueError("Cannot auto-approve generated labels")
    for key in (
        "domains",
        "scenario_actions",
        "voices",
        "difficulties",
        "length_bands",
    ):
        if not cfg[key] or (
            key != "length_bands" and len(cfg[key]) != len(set(cfg[key]))
        ):
            raise ValueError(f"Invalid planning dimension {key}")
    if set(cfg["length_bands"]) - set(LENGTH_LIMITS):
        raise ValueError("Unknown length band")
    if set(cfg["difficulties"]) - set(DIFFICULTY_MAP):
        raise ValueError("Unknown difficulty")
    for primary, contrasts in cfg["contrast_labels"].items():
        if primary not in quotas or len(contrasts) != len(set(contrasts)):
            raise ValueError("Invalid contrast specification")
        if any(x not in quotas or x == primary for x in contrasts):
            raise ValueError("Invalid contrast label")
    if not prompt.strip():
        raise ValueError("Empty generation prompt")
    return cfg, prompt


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def word_count(text):
    return len(WORD_PATTERN.findall(text))


def balanced_choice(pool, global_slot, rng):
    # A coprime stride spreads assignments across thousands of jobs rather
    # than repeating small slot-index patterns. The RNG adds deterministic
    # variation without sacrificing reproducibility.
    return pool[(global_slot + rng.randrange(len(pool))) % len(pool)]


def make_jobs(cfg):
    jobs = []
    size = cfg["examples_per_request"]
    global_slot = 0
    for label, quota in cfg["primary_label_quotas"].items():
        contrasts = cfg["contrast_labels"].get(label, [])
        for index in range(quota // size):
            batch_id = f"{label}_{index:05d}"
            seed = int.from_bytes(
                hashlib.sha256(f"{cfg['seed']}:{batch_id}".encode()).digest()[:8], "big"
            )
            rng = random.Random(seed)
            # Rotate the full pools independently. No position is hard-coded
            # to a single difficulty, length or contrast class.
            options = []
            used_situations = set()
            for slot in range(size):
                # Each domain/scenario combination must be unique inside batch.
                for _ in range(len(cfg["domains"]) * len(cfg["scenario_actions"])):
                    domain = balanced_choice(cfg["domains"], global_slot, rng)
                    scenario = balanced_choice(
                        cfg["scenario_actions"], global_slot + 7, rng
                    )
                    if (domain, scenario) not in used_situations:
                        used_situations.add((domain, scenario))
                        break
                else:
                    raise ValueError(f"Not enough unique scenarios in {batch_id}")
                contrast = (
                    contrasts[(index * size + slot) % len(contrasts)]
                    if contrasts
                    else "none"
                )
                options.append(
                    {
                        "scenario_slot": slot,
                        "domain": domain,
                        "scenario": scenario,
                        "voice": balanced_choice(cfg["voices"], global_slot + 11, rng),
                        "difficulty": DIFFICULTY_MAP[
                            cfg["difficulties"][global_slot % len(cfg["difficulties"])]
                        ],
                        "length": cfg["length_bands"][
                            (global_slot + (global_slot // len(cfg["length_bands"])))
                            % len(cfg["length_bands"])
                        ],
                        "semantic_contrast": contrast,
                    }
                )
                global_slot += 1
            jobs.append(
                {"batch_id": batch_id, "primary_label": label, "slots": options}
            )
    return jobs


def validate(batch, job, cfg):
    examples = batch.examples
    if len(examples) != len(job["slots"]):
        raise ValueError("Incorrect example count")
    allowed = set(cfg["allowed_secondary_labels"]) - {job["primary_label"]}
    seen = set()
    for i, ex in enumerate(examples):
        slot = job["slots"][i]
        if ex.scenario_slot != i or slot["scenario_slot"] != i:
            raise ValueError(f"Slot order mismatch at {i}")
        text = ex.text.strip()
        if (
            not text
            or len(text) > 1200
            or any(ord(ch) < 32 and ch not in "\n\t" for ch in text)
        ):
            raise ValueError(f"Invalid text at slot {i}")
        words = word_count(text)
        low, high = LENGTH_LIMITS[slot["length"]]
        if not low <= words <= high:
            raise ValueError(
                f"Length band violation: slot={i}, band={slot['length']}, words={words}, expected={low}-{high}"
            )
        normalized = normalize(text)
        if normalized in seen:
            raise ValueError(f"In-batch duplicate at slot {i}")
        seen.add(normalized)
        if len(ex.secondary_labels) != len(set(ex.secondary_labels)):
            raise ValueError(f"Duplicate secondary label at slot {i}")
        if not set(ex.secondary_labels).issubset(allowed):
            raise ValueError(f"Unrecognized/primary secondary label at slot {i}")
        # Semantic correctness, naturalness and scenario fidelity cannot be
        # proven by lexical rules. These remain unverified until QA.


def manifest_for(cfg, prompt, jobs):
    return {
        "experiment_id": cfg["experiment_id"],
        "model": cfg["generator_model"],
        "config_sha256": sha(CONFIG.read_bytes()),
        "prompt_sha256": sha(PROMPT.read_bytes()),
        "jobs_sha256": sha(canonical(jobs).encode("utf-8")),
        "batch_count": len(jobs),
        "candidate_target": cfg["candidate_target"],
        "status": "UNAPPROVED_CANDIDATES",
        "generation_only": True,
    }


def request_text(prompt, job):
    slot_targets = []

    for slot in job["slots"]:
        low, high = LENGTH_LIMITS[slot["length"]]
        target = (low + high) // 2

        slot_targets.append(
            {
                "slot": slot["scenario_slot"],
                "length_band": slot["length"],
                "minimum_words": low,
                "maximum_words": high,
                "target_words": target,
            }
        )

    return (
        prompt
        + "\n\nBATCH SPECIFICATION (NOT USER CONTENT):\n"
        + canonical(
            {
                "primary_emotion": job["primary_label"],
                "number_of_comments": len(job["slots"]),
                "slot_assignments": job["slots"],
            }
        )
        + "\n\nMANDATORY WORD-COUNT TARGETS:\n"
        + canonical(slot_targets)
        + "\n\nEvery comment MUST fall within its assigned "
        "minimum and maximum word counts. Aim near the "
        "target word count, not the lower boundary. "
        "Count words in each completed comment before "
        "returning the batch. Revise any comment outside "
        "its band. Preserve natural language and all "
        "other slot constraints."
    )


def read_record(path, job, manifest, cfg):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["batch_id"] != job["batch_id"] or record["manifest_sha256"] != sha(
        canonical(manifest).encode("utf-8")
    ):
        raise ValueError(f"Provenance mismatch: {path}")
    parsed = Batch.model_validate(record["parsed"])
    validate(parsed, job, cfg)
    return record


def rebuild_candidates(root, jobs, manifest, cfg):
    target = root / "candidates.jsonl"
    temp = root / "candidates.jsonl.tmp"
    count = 0
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for job in jobs:
            file = root / "batches" / f"{job['batch_id']}.json"
            if not file.is_file():
                continue
            rec = read_record(file, job, manifest, cfg)
            for item in rec["parsed"]["examples"]:
                row = {
                    "candidate_id": f"syn_large_{job['batch_id']}_{item['scenario_slot']:02d}",
                    "text": item["text"].strip(),
                    "intended_primary_label": job["primary_label"],
                    "proposed_secondary_labels": item["secondary_labels"],
                    "batch_id": job["batch_id"],
                    "scenario_slot": item["scenario_slot"],
                    "response_id": rec["response_id"],
                    "generator_model": rec["returned_model"],
                    "prompt_sha256": manifest["prompt_sha256"],
                    "config_sha256": manifest["config_sha256"],
                    "slot_spec": job["slots"][item["scenario_slot"]],
                    "label_status": "unverified",
                    "approval_status": "pending",
                }
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    os.replace(temp, target)
    return count


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--rebuild", action="store_true")

    parser.add_argument("--max-batches", type=int, default=5)
    parser.add_argument("--spread-labels", action="store_true")
    parser.add_argument("--input-per-million", type=float)
    parser.add_argument("--output-per-million", type=float)

    args = parser.parse_args()

    MAX_ATTEMPTS = 3

    cfg, prompt = load_spec()
    jobs = make_jobs(cfg)
    manifest = manifest_for(cfg, prompt, jobs)

    root = ROOT / cfg["output_root"]
    batch_dir = root / "batches"
    rejected_dir = root / "rejected_batches"
    manifest_path = root / "generation_manifest.json"

    paths = [batch_dir / f"{job['batch_id']}.json" for job in jobs]

    completed = [(job, path) for job, path in zip(jobs, paths) if path.is_file()]

    print(f"Planned: {len(jobs)} batches / {manifest['candidate_target']} candidates")
    print(f"Completed: {len(completed)} batches")
    print(f"Generator: {manifest['model']}")

    if args.plan:
        print("Planning dimensions and constraints:")

        for key in ("difficulty", "length", "voice"):
            counts = Counter(slot[key] for job in jobs for slot in job["slots"])
            print(key, dict(sorted(counts.items())))

        print(
            "Unique slots per batch:",
            all(
                len({(s["domain"], s["scenario"]) for s in job["slots"]})
                == len(job["slots"])
                for job in jobs
            ),
        )
        print(
            "Manifest SHA256:",
            sha(canonical(manifest).encode("utf-8")),
        )
        print("No API requests made. No files created.")
        return

    # Preserve existing generation artifacts.
    root.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if saved_manifest != manifest:
            raise RuntimeError(
                "Generation manifest mismatch. Existing campaign must be preserved."
            )

    else:
        existing_batches = list(batch_dir.iterdir()) if batch_dir.exists() else []

        if existing_batches or (root / "candidates.jsonl").exists():
            raise RuntimeError("Generation artifacts exist without a manifest.")

        with manifest_path.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2)
            stream.write("\n")

    batch_dir.mkdir(parents=True, exist_ok=True)

    for job, path in completed:
        read_record(path, job, manifest, cfg)

    if args.rebuild:
        count = rebuild_candidates(root, jobs, manifest, cfg)
        print("Rebuilt candidate count:", count)
        return

    if args.max_batches <= 0:
        raise ValueError("--max-batches must be positive")

    if cfg["generator_model"].startswith("SET_"):
        raise ValueError("Configure a real generator model ID")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY missing")

    prices = (
        args.input_per_million,
        args.output_per_million,
    )

    if any(p is None or p <= 0 for p in prices):
        raise ValueError(
            "Specify positive input/output token prices for cost reporting."
        )

    # Read previously rejected attempts.
    rejected_dir.mkdir(parents=True, exist_ok=True)

    attempts_by_batch = Counter()
    total_usd = 0.0

    def usage_cost(usage):
        usage = usage or {}
        return (
            usage.get("input_tokens", 0) * prices[0]
            + usage.get("output_tokens", 0) * prices[1]
        ) / 1_000_000

    for _, path in completed:
        record = json.loads(path.read_text(encoding="utf-8"))
        total_usd += usage_cost(record.get("usage"))

    for path in rejected_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))

        batch_id = record["batch_id"]
        attempts_by_batch[batch_id] += 1
        total_usd += usage_cost(record.get("usage"))

    remaining = [(job, path) for job, path in zip(jobs, paths) if not path.exists()]

    if args.spread_labels:
        remaining.sort(
            key=lambda item: (
                int(item[0]["batch_id"].rsplit("_", 1)[1]),
                item[0]["primary_label"],
            )
        )

    from openai import OpenAI

    # Exactly three application-level attempts maximum.
    # Avoid hidden SDK retries multiplying API calls.
    client = OpenAI(max_retries=0, timeout=150)

    succeeded = 0
    exhausted = 0
    attempted_batches = 0

    for job, path in remaining[: args.max_batches]:
        batch_id = job["batch_id"]
        previous_attempts = attempts_by_batch[batch_id]

        if previous_attempts >= MAX_ATTEMPTS:
            print(
                f"[SKIPPED] {batch_id}: {previous_attempts} previous rejected attempts"
            )
            exhausted += 1
            continue

        attempted_batches += 1
        request = request_text(prompt, job)
        request_hash = sha(request.encode("utf-8"))

        success = False

        for attempt in range(
            previous_attempts + 1,
            MAX_ATTEMPTS + 1,
        ):
            # API and transport errors are deliberately fatal.
            # Validation failures below are recoverable.

            try:
                response = client.responses.parse(
                    model=cfg["generator_model"],
                    input=[
                        {
                            "role": "user",
                            "content": request,
                        }
                    ],
                    text_format=Batch,
                    store=False,
                    max_output_tokens=8000,
                )

            except ValidationError as error:
                # The SDK could not parse the structured response.
                # No parsed Response object is available here.
                # Record the failure before retrying.
                rejected_record = {
                    "batch_id": batch_id,
                    "attempt": attempt,
                    "status": "REJECTED_PARSE",
                    "response_id": None,
                    "returned_model": None,
                    "request_sha256": request_hash,
                    "validation_error": str(error),
                    "usage": None,
                    "parsed": None,
                }

                failed_path = rejected_dir / f"{batch_id}_parse_attempt_{attempt}.json"

                with failed_path.open(
                    "x",
                    encoding="utf-8",
                ) as stream:
                    json.dump(
                        rejected_record,
                        stream,
                        ensure_ascii=False,
                        indent=2,
                    )
                    stream.write("\n")

                attempts_by_batch[batch_id] += 1

                print(f"[PARSE FAILED] {batch_id} attempt {attempt}/{MAX_ATTEMPTS}")
                print(f"[SAVED] {failed_path}")
                print(f"[REASON] {error}")

                # Continue the existing three-attempt loop.
                continue

            parsed = response.output_parsed
            usage = response.usage.model_dump(mode="json") if response.usage else None

            cost = usage_cost(usage)
            total_usd += cost

            try:
                if parsed is None:
                    raise ValueError("Missing parsed response")

                validate(parsed, job, cfg)

            except ValueError as error:
                rejected_record = {
                    "batch_id": batch_id,
                    "attempt": attempt,
                    "status": "REJECTED",
                    "response_id": response.id,
                    "returned_model": response.model,
                    "request_sha256": request_hash,
                    "validation_error": str(error),
                    "usage": usage,
                    "parsed": (
                        parsed.model_dump(mode="json") if parsed is not None else None
                    ),
                }

                failed_path = rejected_dir / f"{batch_id}_{response.id}.json"

                with failed_path.open("x", encoding="utf-8") as stream:
                    json.dump(
                        rejected_record,
                        stream,
                        ensure_ascii=False,
                        indent=2,
                    )
                    stream.write("\n")

                attempts_by_batch[batch_id] += 1

                print(
                    f"[REJECTED] {batch_id} attempt {attempt}/{MAX_ATTEMPTS}: {error}"
                )
                print(f"[SAVED] {failed_path}")
                print(f"Estimated campaign USD: {total_usd:.4f}")

                continue

            # Successful batch: persist before marking complete.
            record = {
                "batch_id": batch_id,
                "manifest_sha256": sha(canonical(manifest).encode("utf-8")),
                "request_sha256": request_hash,
                "response_id": response.id,
                "returned_model": response.model,
                "usage": usage,
                "parsed": parsed.model_dump(mode="json"),
            }

            temp = path.with_suffix(".json.tmp")

            with temp.open("x", encoding="utf-8") as stream:
                json.dump(
                    record,
                    stream,
                    ensure_ascii=False,
                )
                stream.write("\n")

            os.replace(temp, path)

            succeeded += 1
            success = True

            print(
                f"[OK] {batch_id} "
                f"attempt={attempt}/{MAX_ATTEMPTS} "
                f"| new batches={succeeded} "
                f"| estimated campaign USD={total_usd:.4f}"
            )

            break

        if not success:
            exhausted += 1
            print(f"[EXHAUSTED] {batch_id}: maximum {MAX_ATTEMPTS} attempts reached")

    count = rebuild_candidates(root, jobs, manifest, cfg)

    print("\nGENERATION SUMMARY")
    print("Scheduled batches:", len(remaining[: args.max_batches]))
    print("Batches attempted:", attempted_batches)
    print("Successful batches:", succeeded)
    print("Exhausted/skipped batches:", exhausted)
    print("Rebuilt candidate count:", count)
    print(f"Recorded estimated campaign USD: ${total_usd:.4f}")
    print("Spending limit: DISABLED")
    print("Maximum attempts per batch:", MAX_ATTEMPTS)
    print("Candidates remain UNAPPROVED.")


if __name__ == "__main__":
    main()
