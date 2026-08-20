from pathlib import Path

import yaml


CONFIG_PATH = Path("ml/configs/dair_ai_emotion.yaml")


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_dair_config_has_verified_split_contract() -> None:
    config = load_config()

    assert config["hf_id"] == "dair-ai/emotion"
    assert config["hf_config"] == "split"
    assert config["splits"] == {
        "train": "train",
        "validation": "validation",
        "test": "test",
    }
    assert config["expected_split_sizes"] == {
        "train": 16000,
        "validation": 2000,
        "test": 2000,
    }


def test_dair_config_has_verified_label_order() -> None:
    config = load_config()

    assert config["labels"] == [
        "sadness",
        "joy",
        "love",
        "anger",
        "fear",
        "surprise",
    ]
    assert config["label_ids"] == {
        "sadness": 0,
        "joy": 1,
        "love": 2,
        "anger": 3,
        "fear": 4,
        "surprise": 5,
    }
