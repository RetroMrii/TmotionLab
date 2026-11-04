"""Read-only P12 verification. Never reads the GoEmotions test split or writes research files."""
import json
from itertools import islice
from backend.app.services.goemotions import GoEmotionsPredictor, MODEL_ID, ROOT, verify_artifacts


def main():
    labels = verify_artifacts()
    predictor = GoEmotionsPredictor()
    with (ROOT / "data/processed/goemotions_v001/validation.jsonl").open(encoding="utf-8") as stream:
        source = [json.loads(line) for line in islice(stream, 16)]
    with (ROOT / f"ml/artifacts/predictions/{MODEL_ID}_validation.jsonl").open(encoding="utf-8") as stream:
        saved = [json.loads(line) for line in islice(stream, 16)]
    if len(source) != 16 or len(saved) != 16:
        raise ValueError("Missing validation fixture rows")
    differences = []
    for row, expected in zip(source, saved, strict=True):
        if row["example_id"] != expected["example_id"]:
            raise ValueError("Validation fixture alignment mismatch")
        response = predictor.predict(row["text"])
        differences.extend(abs(item["score"] - p) for item, p in zip(response["scores"], expected["probabilities"], strict=True))
    maximum = max(differences)
    if maximum > 0.002:
        raise ValueError(f"Validation probability mismatch: {maximum}")
    print(json.dumps({"model_id": MODEL_ID, "frozen_hashes_verified": True,
                      "label_count": len(labels), "device": predictor.device,
                      "validation_examples": 16, "maximum_probability_difference": maximum,
                      "tolerance": 0.002, "test_split_read": False}, indent=2))


if __name__ == "__main__":
    main()
