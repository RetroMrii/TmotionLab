import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "data/synthetic/goemotions_large_scale/quality/fixture_candidates.jsonl"

OUTPUT = (
    ROOT / "data/synthetic/goemotions_large_scale/quality/"
    "semantic_control_candidates.jsonl"
)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Preserve the previous fixture: {OUTPUT}")

    rows = [
        json.loads(line)
        for line in SOURCE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    if len(rows) != 4:
        raise ValueError(f"Expected four original fixture rows, got {len(rows)}")

    original = rows[0].copy()

    duplicate = original.copy()
    duplicate["candidate_id"] = "control_exact_duplicate"

    paraphrase = original.copy()
    paraphrase["candidate_id"] = "control_paraphrase"
    paraphrase["text"] = (
        "I thought my keys were gone, but they were "
        "sitting in my other bag the whole time."
    )

    rows.extend([duplicate, paraphrase])

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open(
        "x",
        encoding="utf-8",
        newline="\n",
    ) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("OPT-009 — SEMANTIC POSITIVE CONTROLS")
    print("Original candidates: 4")
    print("Added exact duplicate: 1")
    print("Added paraphrase: 1")
    print("Total: 6")
    print("Saved:", OUTPUT)


if __name__ == "__main__":
    main()
