import pytest

from ml.scripts.generate_goemotions_synthetic_v002 import (
    GeneratedBatch,
    build_jobs,
    load_spec,
    validate_batch,
)


def test_generation_plan():
    config, _ = load_spec()
    jobs = build_jobs(config)

    assert len(jobs) == 60
    assert sum(j["label"] == "grief" for j in jobs) == 15

    for label in config["targets"]:
        themes = [
            theme for job in jobs if job["label"] == label for theme in job["scenarios"]
        ]
        assert len(themes) == 75
        assert len(set(themes)) == 15


def test_valid_batch():
    config, _ = load_spec()
    job = build_jobs(config)[0]

    batch = GeneratedBatch.model_validate(
        {
            "examples": [
                {
                    "scenario_slot": i,
                    "text": f"A distinct, meaningful loss affected me "
                    f"in situation number {i} today.",
                    "secondary_labels": ["sadness"],
                }
                for i in range(5)
            ]
        }
    )

    validate_batch(batch, job)


def test_reject_invalid_slots():
    config, _ = load_spec()
    job = build_jobs(config)[0]

    batch = GeneratedBatch.model_validate(
        {
            "examples": [
                {
                    "scenario_slot": 0,
                    "text": f"A different meaningful experience "
                    f"in situation number {i} today.",
                    "secondary_labels": [],
                }
                for i in range(5)
            ]
        }
    )

    with pytest.raises(ValueError):
        validate_batch(batch, job)
