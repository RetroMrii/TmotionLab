import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

from openai import OpenAI

from ml.scripts.generate_goemotions_synthetic_pilot import (
    GeneratedBatch,
    build_prompt,
    load_settings,
    normalize,
    validate_examples,
)


ROOT = Path(__file__).resolve().parents[2]

COMPARISON_CONFIG = ROOT / "ml/configs/synthetic_generator_comparison_v001.json"

OUTPUT_DIR = ROOT / "data/synthetic/generator_comparison_v001"

PRICES = {
    "gpt-6.1-terra": (1.00, 5.00),
    "gpt-6-luna": (0.10, 0.50),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_json_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as stream:
        json.dump(
            data,
            stream,
            ensure_ascii=False,
            indent=2,
        )
        stream.write("\n")


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def plan_jobs(comparison):
    jobs = []

    for label in comparison["targets"]:
        for model in comparison["models"]:
            jobs.append(
                {
                    "model": model,
                    "label": label,
                    "batch_id": f"{label}_000",
                    "count": comparison["examples_per_request"],
                }
            )

    return jobs


def record_path(job):
    return OUTPUT_DIR / job["model"] / f"{job['batch_id']}.json"


def check_spec(comparison, base_config):
    if comparison["experiment_id"] != ("synthetic_generator_comparison_v001"):
        raise ValueError("Unexpected experiment ID")

    if set(comparison["models"]) != set(PRICES):
        raise ValueError("Unexpected model selection")

    if set(comparison["targets"]) != set(base_config["targets"]):
        raise ValueError("Target-label mismatch")

    if (
        comparison["examples_per_request"] != 5
        or comparison["examples_per_label_per_model"] != 5
    ):
        raise ValueError("Unexpected comparison size")

    if comparison["reasoning_effort"] != "none":
        raise ValueError("Unexpected reasoning setting")

    if comparison["prompt_version"] != (base_config["prompt_version"]):
        raise ValueError("Prompt version mismatch")


def summarize(records, jobs):
    totals = {}

    for model in PRICES:
        subset = [record for record in records if record["requested_model"] == model]

        input_tokens = sum(item["usage"]["input_tokens"] for item in subset)

        output_tokens = sum(item["usage"]["output_tokens"] for item in subset)

        input_rate, output_rate = PRICES[model]

        upper_estimate = (
            input_tokens * input_rate + output_tokens * output_rate
        ) / 1_000_000

        counts = Counter(
            item["primary_label"] for record in subset for item in record["candidates"]
        )

        texts = [
            normalize(item["text"])
            for record in subset
            for item in record["candidates"]
        ]

        totals[model] = {
            "completed_batches": len(subset),
            "candidate_count": len(texts),
            "label_counts": dict(counts),
            "unique_normalized_texts": len(set(texts)),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd_before_discounts": (round(upper_estimate, 6)),
        }

    return totals


def main():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--plan",
        action="store_true",
    )

    group.add_argument(
        "--run",
        action="store_true",
    )

    args = parser.parse_args()

    comparison = load_json(COMPARISON_CONFIG)
    base_config, base_prompt = load_settings()

    check_spec(comparison, base_config)

    jobs = plan_jobs(comparison)

    manifest = {
        "experiment_id": comparison["experiment_id"],
        "comparison_config_sha256": sha256(COMPARISON_CONFIG),
        "base_config_sha256": sha256(
            ROOT / "ml/configs/synthetic_goemotions_pilot_v001.json"
        ),
        "prompt_sha256": sha256(
            ROOT / "ml/configs/synthetic/synthetic_emotion_generation_v001.txt"
        ),
        "planned_batches": len(jobs),
        "planned_candidates": sum(job["count"] for job in jobs),
        "status": "UNAPPROVED",
    }

    if args.plan:
        print(json.dumps(manifest, indent=2))
        print("\nPlanned jobs:")

        for job in jobs:
            print(
                job["model"],
                job["label"],
                job["count"],
            )

        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = OUTPUT_DIR / "manifest.json"

    if manifest_path.exists():
        if load_json(manifest_path) != manifest:
            raise ValueError("Comparison specification changed")
    else:
        if any(OUTPUT_DIR.rglob("*.json")):
            raise RuntimeError("Existing records without manifest")

        save_json_once(manifest_path, manifest)

    client = OpenAI(
        max_retries=2,
        timeout=120,
    )

    for job in jobs:
        path = record_path(job)

        if path.exists():
            record = load_json(path)

            if (
                record["requested_model"] != job["model"]
                or record["batch_id"] != job["batch_id"]
                or record["primary_label"] != job["label"]
                or record["manifest"] != manifest
            ):
                raise ValueError(f"Saved record mismatch: {path}")

            continue

        # Use the same original generation prompt for
        # both models. No held-out examples are included.
        prompt_job = {
            "label": job["label"],
            "count": job["count"],
            "batch_id": job["batch_id"],
            "ordinal": 0,
        }

        prompt = build_prompt(
            base_prompt,
            base_config,
            prompt_job,
        )

        response = client.responses.parse(
            model=job["model"],
            reasoning={"effort": comparison["reasoning_effort"]},
            input=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            text_format=GeneratedBatch,
            store=False,
        )

        if response.output_parsed is None:
            raise RuntimeError(f"No parsed response: {job}")

        validate_examples(
            response.output_parsed,
            prompt_job,
            base_config,
        )

        usage = response.usage

        if usage is None:
            raise RuntimeError("Missing API usage metadata")

        candidates = []

        for index, example in enumerate(response.output_parsed.examples):
            candidates.append(
                {
                    "candidate_id": (f"{job['model']}_{job['batch_id']}_{index:02d}"),
                    "primary_label": job["label"],
                    "text": example.text.strip(),
                    "secondary_labels": (example.secondary_labels),
                    "approval_status": "pending",
                }
            )

        record = {
            "manifest": manifest,
            "requested_model": job["model"],
            "returned_model": response.model,
            "batch_id": job["batch_id"],
            "primary_label": job["label"],
            "response_id": response.id,
            "request_prompt": prompt,
            "usage": {
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            },
            "candidates": candidates,
            "raw_response": response.model_dump(mode="json"),
        }

        save_json_once(path, record)

        print(
            "[OK]",
            job["model"],
            job["label"],
            f"{usage.input_tokens} input tokens,",
            f"{usage.output_tokens} output tokens",
        )

    records = [load_json(record_path(job)) for job in jobs]

    summary = {
        "experiment_id": (comparison["experiment_id"]),
        "model_results": summarize(
            records,
            jobs,
        ),
        "semantic_quality_evaluated": False,
        "training_approved": False,
    }

    summary_path = OUTPUT_DIR / "summary.json"

    if summary_path.exists():
        if load_json(summary_path) != summary:
            raise ValueError("Existing summary differs")
    else:
        save_json_once(
            summary_path,
            summary,
        )

    print("\nCOMPARISON SUMMARY")
    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\n[OK] Generator comparison completed")


if __name__ == "__main__":
    main()
