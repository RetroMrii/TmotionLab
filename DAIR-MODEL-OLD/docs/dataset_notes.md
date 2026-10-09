# TmotionLab Dataset Notes

## DAIR.AI Emotion — initial single-label dataset

**Verification date:** 2026-08-20  
**Hugging Face dataset:** `dair-ai/emotion`  
**Selected configuration:** `split`  
**Pinned repository revision:** `cab853a1dbdf4c42c2b3ef2173804746df8825fe`

### Verified contract

| Field | Verified value |
|---|---|
| Language | English |
| Task | Single-label multi-class text classification |
| Text column | `text` |
| Label column | `label` (`ClassLabel`) |
| Train | 16,000 rows |
| Validation | 2,000 rows |
| Test | 2,000 rows |
| Total | 20,000 rows |

### Label mapping

| ID | Label |
|---:|---|
| 0 | sadness |
| 1 | joy |
| 2 | love |
| 3 | anger |
| 4 | fear |
| 5 | surprise |

### Source and usage notes

The Hugging Face dataset card identifies the license as `other`. An earlier dataset loader and the DAIR.AI repository state that the dataset should be used for educational and research purposes. TmotionLab therefore records that wording directly instead of treating the dataset as if it had a standard permissive software/data license.

The dataset card describes the samples as English Twitter messages and leaves several documentation sections incomplete, including detailed collection, annotation, bias, and sensitive-information notes. These documentation gaps must be reflected later in the model card and public limitations.

### Reproducibility policy

TmotionLab uses the provided `split` configuration rather than creating a new random split. The pipeline must preserve the upstream train/validation/test boundaries so every model is compared on the identical held-out examples.

Before preparation or training, run:

```bash
python ml/scripts/inspect_dataset.py --config ml/configs/dair_ai_emotion.yaml --output ml/artifacts/metrics/dair_ai_emotion_inspection.json
```

The inspection script fails if the expected splits, columns, split sizes, or label order change.

### Still to measure locally

The upstream metadata verifies the schema, but the following are intentionally left for the inspection/preparation pipeline rather than guessed from documentation:

- exact class counts in each split;
- duplicate text frequency within/across splits;
- text-length distribution;
- empty/null text count;
- representative edge cases;
- preprocessing consequences.
