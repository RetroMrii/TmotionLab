from pathlib import Path


def test_processed_dataset_directory_target():
    path = Path("data/processed/dair_ai_emotion")

    assert path.parent.name == "processed"
    assert path.name == "dair_ai_emotion"


def test_expected_processed_files():
    expected = {
        "train.csv",
        "validation.csv",
        "test.csv",
        "metadata.json",
    }

    assert expected == {
        "train.csv",
        "validation.csv",
        "test.csv",
        "metadata.json",
    }
