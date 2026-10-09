from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_ID = "google-research-datasets/go_emotions"
CONFIG = "simplified"

EXPECTED_SPLITS = {
    "train": 43410,
    "validation": 5426,
    "test": 5427,
}

EXPECTED_LABEL_COUNT = 28


def main() -> None:
    revision = HfApi().dataset_info(DATASET_ID).sha
    assert revision, "Could not resolve dataset revision"

    print(f"Dataset: {DATASET_ID}")
    print(f"Config: {CONFIG}")
    print(f"Revision: {revision}")

    dataset = load_dataset(
        DATASET_ID,
        CONFIG,
        revision=revision,
    )

    assert set(dataset.keys()) == set(EXPECTED_SPLITS)

    reference_names = None
    all_ids = set()

    for split_name, expected_count in EXPECTED_SPLITS.items():
        split = dataset[split_name]
        features = split.features

        assert len(split) == expected_count, (
            f"{split_name}: expected {expected_count}, got {len(split)}"
        )

        assert "text" in features
        assert "labels" in features

        assert features["labels"].feature.num_classes == (EXPECTED_LABEL_COUNT)

        label_names = features["labels"].feature.names

        if reference_names is None:
            reference_names = label_names
        else:
            assert label_names == reference_names

        id_field = (
            "id"
            if "id" in features
            else "comment_id"
            if "comment_id" in features
            else None
        )
        assert id_field is not None, f"{split_name}: no recognized source ID field"

        split_ids = set()

        for row in split:
            assert isinstance(row["text"], str)
            assert row["text"].strip()
            assert row["labels"]

            assert all(
                isinstance(label, int) and 0 <= label < EXPECTED_LABEL_COUNT
                for label in row["labels"]
            )

            assert len(row["labels"]) == len(set(row["labels"]))

            source_id = row[id_field]
            assert isinstance(source_id, str)
            assert source_id

            assert source_id not in split_ids, (
                f"Duplicate source ID in {split_name}: {source_id}"
            )
            split_ids.add(source_id)

        overlap = all_ids.intersection(split_ids)
        assert not overlap, f"Cross-split source ID overlap: {len(overlap)}"

        all_ids.update(split_ids)

        multi_label_count = sum(len(row["labels"]) > 1 for row in split)

        print(
            f"{split_name}: {len(split)} rows; "
            f"multi-label rows: {multi_label_count}; "
            f"ID field: {id_field}"
        )

    print("\nVerified label mapping:")

    for index, name in enumerate(reference_names):
        print(f"{index:02d}: {name}")

    assert reference_names[27] == "neutral"

    print("\n[OK] GoEmotions schema verification passed")


if __name__ == "__main__":
    main()
