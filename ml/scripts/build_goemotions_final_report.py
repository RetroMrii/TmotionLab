import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports/goemotions"

SOURCES = {
    "robustness": REPORT_DIR / "multiseed_robustness_v001.json",
    "rare": REPORT_DIR / "multiseed_rare_label_analysis_v001.json",
    "test": REPORT_DIR / "final_heldout_test_v001.json",
    "freeze": REPORT_DIR / "final_model_freeze_v001.json",
}

OUTPUT = REPORT_DIR / "goemotions_research_final_v001.md"

STRATEGIES = ("uniform", "rare4x", "synthetic", "hybrid")
SEEDS = (42, 123, 456)
TARGETS = ("grief", "nervousness", "pride", "relief")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    return f"{float(value):.4f}"


def metric_row(strategy, robustness):
    stats = robustness["summary"][strategy]

    def mean(name):
        return fmt(stats[name]["mean"])

    def std(name):
        return fmt(stats[name]["sample_std"])

    return (
        f"| {strategy} | "
        f"{mean('macro_f1')} ± {std('macro_f1')} | "
        f"{mean('macro_average_precision')} ± "
        f"{std('macro_average_precision')} | "
        f"{mean('micro_f1')} |"
    )


def check_sources(data):
    robustness = data["robustness"]
    rare = data["rare"]
    test = data["test"]
    freeze = data["freeze"]

    if (
        robustness["report_id"] != "multiseed_robustness_v001"
        or robustness["evaluation_split"] != "validation"
        or robustness["test_evaluated"] is not False
        or tuple(robustness["seeds"]) != SEEDS
        or set(robustness["strategies"]) != set(STRATEGIES)
    ):
        raise ValueError("Robustness report mismatch")

    if (
        rare["analysis_id"] != "multiseed_rare_label_analysis_v001"
        or rare["split"] != "validation"
        or rare["test_evaluated"] is not False
        or tuple(rare["seeds"]) != SEEDS
        or tuple(rare["target_labels"]) != TARGETS
    ):
        raise ValueError("Rare-label analysis mismatch")

    for strategy in STRATEGIES:
        for seed in SEEDS:
            if str(seed) not in robustness["runs"][strategy]:
                raise ValueError(f"Missing robustness run: {strategy}/{seed}")
            if str(seed) not in rare["runs"][strategy]:
                raise ValueError(f"Missing rare-label run: {strategy}/{seed}")

            a = robustness["runs"][strategy][str(seed)]
            b = rare["runs"][strategy][str(seed)]

            if (
                a["experiment_id"] != b["experiment_id"]
                or a["metrics_sha256"] != b["metrics_sha256"]
                or a["predictions_sha256"] != b["prediction_sha256"]
            ):
                raise ValueError(f"Cross-report artifact mismatch: {strategy}/{seed}")

    if (
        freeze["freeze_id"] != "final_model_freeze_v001"
        or freeze["primary_experiment"] != "roberta_goemotions_rare4x_v001"
        or freeze["seed"] != 42
        or float(freeze["classification_threshold"]) != 0.25
        or freeze["test_time_tuning_allowed"] is not False
    ):
        raise ValueError("Frozen selection mismatch")

    if (
        test["report_id"] != "final_heldout_test_v001"
        or test["experiment_id"] != freeze["primary_experiment"]
        or test["split"] != "test"
        or test["examples"] != 5427
        or float(test["threshold"]) != 0.25
        or test["test_time_tuning_performed"] is not False
        or test["freeze_sha256"] != sha256(SOURCES["freeze"])
    ):
        raise ValueError("Held-out evaluation mismatch")


def build_report(data):
    robustness = data["robustness"]
    rare = data["rare"]
    test = data["test"]

    lines = [
        "# TmotionLab — GoEmotions Research Report",
        "",
        "**Study:** P11 — Rare-Emotion Sampling and Synthetic Augmentation",
        "",
        "**Status:** Completed exploratory research; "
        "primary model evaluated on held-out test.",
        "",
        "## Executive summary",
        "",
        "This study investigated whether targeted synthetic "
        "examples improve recognition of rare emotions "
        "in multi-label text classification, relative to "
        "real-data oversampling.",
        "",
        "Across three training seeds and four sampling "
        "strategies, all rare-label interventions improved "
        "validation macro F1 relative to uniform sampling.",
        "",
        "Hybrid sampling achieved the highest mean "
        "validation macro F1, but its advantage over "
        "real-data oversampling was too small to "
        "establish reliable superiority.",
        "",
        "Real 4× oversampling was selected as the primary "
        "model under the predefined pragmatic decision: "
        "prefer the simpler human-annotated-data approach "
        "unless hybrid demonstrates a meaningful benefit.",
        "",
        "## 1. Dataset and task",
        "",
        "- Dataset: GoEmotions simplified.",
        "- Task: 28-label multi-label emotion classification.",
        "- Processed training rows: 43,252.",
        "- Validation rows: 5,426.",
        "- Held-out test rows: 5,427.",
        "- Training/validation normalized-text overlap "
        "was addressed during dataset preparation.",
        "- The official test labels were excluded from "
        "training, model selection and threshold tuning.",
        "",
        "## 2. Baseline and interventions",
        "",
        "| Strategy | Training intervention |",
        "|---|---|",
        "| Uniform | Real examples sampled uniformly with replacement |",
        "| Real 4× | Increased sampling weight for "
        "examples containing targeted rare emotions |",
        "| Synthetic | Targeted draws from 299 generated "
        "examples mixed with real data |",
        "| Hybrid | 50/50 mixture of real 4× and synthetic sampling schedules |",
        "",
        "The synthetic candidates were produced using "
        "GPT-6.1 Terra. The original pilot contained "
        "300 candidates; one confirmed near-duplicate "
        "was excluded.",
        "",
        "Individual human semantic annotation was waived "
        "for this dataset as a one-time research decision. "
        "Generator-proposed secondary labels were "
        "therefore not independently verified.",
        "",
        "## 3. Experimental controls",
        "",
        "- Base model: RoBERTa-base, pinned source revision.",
        "- Seeds: 42, 123 and 456.",
        "- Training: 4 epochs and 43,252 draws per epoch.",
        "- Optimization: 5,408 steps per run.",
        "- Checkpoint selection: validation macro AP.",
        "- Global classification threshold: selected "
        "from a fixed validation grid per run.",
        "- Sampling schedules: deterministic and audited.",
        "- Test split: excluded from strategy development.",
        "",
        "## 4. Three-seed validation results",
        "",
        "| Strategy | Macro F1 mean ± SD | Macro AP mean ± SD | Micro F1 mean |",
        "|---|---:|---:|---:|",
    ]

    for strategy in STRATEGIES:
        lines.append(metric_row(strategy, robustness))

    lines.extend(
        [
            "",
            "Statistics are descriptive. Three seeds do not "
            "establish statistical significance or reliable "
            "superiority between closely performing methods.",
            "",
            "## 5. Four-target rare-emotion analysis",
            "",
            "| Strategy | Four-target mean F1 ± SD | Four-target mean AP |",
            "|---|---:|---:|",
        ]
    )

    for strategy in STRATEGIES:
        stats = rare["summary"][strategy]
        f1 = stats["rare_macro_f1"]
        ap = stats["rare_macro_ap"]

        lines.append(
            f"| {strategy} | {fmt(f1['mean'])} ± "
            f"{fmt(f1['sample_std'])} | "
            f"{fmt(ap['mean'])} |"
        )

    lines.extend(
        [
            "",
            "### Per-label mean validation F1",
            "",
            "| Emotion | Uniform | Real 4× | Synthetic | Hybrid |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    for label in TARGETS:
        values = [
            fmt(rare["summary"][strategy]["per_label"][label]["f1"]["mean"])
            for strategy in STRATEGIES
        ]
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend(
        [
            "",
            "Hybrid improved nervousness F1 relative to "
            "real 4× in all three seeds, but its differences "
            "on the other rare labels were mixed.",
            "",
            "The rare-label evaluation supports are small; "
            "conclusions about individual emotions therefore "
            "remain uncertain.",
            "",
            "## 6. Final model selection",
            "",
            "- Selected: `roberta_goemotions_rare4x_v001`.",
            "- Seed: 42 (fixed reference seed).",
            "- Threshold: 0.25, chosen on validation.",
            "- Selection rationale: competitive performance "
            "without requiring unverified synthetic labels.",
            "- Hybrid retained as the strongest research "
            "alternative, not the primary deployed checkpoint.",
            "",
            "## 7. Held-out test results",
            "",
            "| Metric | Score |",
            "|---|---:|",
        ]
    )

    for label, key in (
        ("Macro F1", "macro_f1"),
        ("Micro F1", "micro_f1"),
        ("Macro AP", "macro_average_precision"),
        ("Exact-match accuracy", "exact_match_accuracy"),
        ("Example-based F1", "example_based_f1"),
    ):
        lines.append(f"| {label} | {fmt(test[key])} |")

    lines.extend(
        [
            "",
            f"Test examples: **{test['examples']:,}**.",
            "",
            "### Rare-label held-out performance",
            "",
            "| Label | F1 | Support |",
            "|---|---:|---:|",
        ]
    )

    for label in TARGETS:
        result = test["per_label"][label]
        lines.append(f"| {label} | {fmt(result['f1'])} | {result['support']} |")

    lines.extend(
        [
            "",
            "The final test was used once after freezing "
            "the primary model and threshold. "
            "No test-time tuning was performed.",
            "",
            "## 8. Limitations",
            "",
            "1. Only three paired training seeds were used.",
            "2. Rare-label validation and test supports are small.",
            "3. Synthetic labels were not independently reviewed.",
            "4. Thresholds and interventions were developed "
            "using the same validation split.",
            "5. Models may behave differently on informal "
            "real-world inputs outside the benchmark.",
            "6. The held-out evaluation covers the selected "
            "real 4× model, not all four strategies.",
            "",
            "## 9. Conclusions",
            "",
            "Targeted exposure to rare-emotion examples "
            "consistently improved validation macro F1. "
            "Synthetic generation supplied useful training "
            "signal, particularly for nervousness, but did "
            "not demonstrate reliable overall superiority "
            "to real-data oversampling.",
            "",
            "The selected real 4× model obtained a held-out "
            f"macro F1 of {fmt(test['macro_f1'])}, with "
            f"micro F1 {fmt(test['micro_f1'])} and "
            f"exact-match accuracy "
            f"{fmt(test['exact_match_accuracy'])}.",
            "",
            "The project can now transition to artifact "
            "retention and application integration without "
            "altering the completed research results.",
            "",
            "## 10. Evidence and reproducibility",
            "",
            "The following source checksums identify the "
            "exact results used to construct this report:",
            "",
            "| Source | SHA-256 |",
            "|---|---|",
        ]
    )

    for name, path in SOURCES.items():
        lines.append(f"| `{path.relative_to(ROOT)}` | `{sha256(path)}` |")

    lines.extend(
        [
            "",
            "The source artifacts additionally retain per-run "
            "metrics, prediction hashes, selected thresholds, "
            "sampling audits and experiment provenance.",
            "",
        ]
    )

    return "\n".join(lines)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Research report already exists: {OUTPUT}")

    data = {name: read_json(path) for name, path in SOURCES.items()}

    check_sources(data)

    report = build_report(data)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(report)

    print("P11-F2 — FINAL RESEARCH REPORT")
    print("-" * 48)
    print("Training experiments verified: 12")
    print("Held-out test examples:", data["test"]["examples"])
    print("Selected model: roberta_goemotions_rare4x_v001")
    print("Research report:", OUTPUT)
    print("[OK] Final research report generated")


if __name__ == "__main__":
    main()
