# TmotionLab Model Comparison

## Purpose

This report compares the three promoted TmotionLab emotion-classification models on the same 2,000-example DAIR.AI Emotion test split.

The comparison is based on identical `example_id` values, text, and ground-truth labels across all model prediction artifacts.

The three models are:

1. TF-IDF + Logistic Regression baseline.
2. From-scratch NumPy MLP using TF-IDF input features.
3. PyTorch TextCNN using learned embeddings and convolutional text features.

## Test Metrics

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.8320 | 0.7301 | 0.8208 |
| NumPy MLP v002 | 0.8770 | 0.8058 | 0.8742 |
| TextCNN v001 | 0.8965 | 0.8510 | 0.8967 |

TextCNN improves over the Logistic Regression baseline by:

- **Accuracy:** +0.0645
- **Macro F1:** +0.1209
- **Weighted F1:** +0.0759

TextCNN also improves macro F1 over the NumPy MLP by **+0.0451**.

## Agreement Analysis

Across the 2000 aligned test examples:

- all three models agree on **82.65%** of examples;
- Logistic Regression and NumPy MLP agree on **89.25%**;
- Logistic Regression and TextCNN agree on **85.15%**;
- NumPy MLP and TextCNN agree on **89.85%**;
- all three models produce different predictions on **21** examples.

High agreement does not imply correctness. There are **93** examples where all three models make the same incorrect prediction.

## TextCNN Gains Over the Baseline

TextCNN corrects **205** examples that the Logistic Regression baseline predicts incorrectly.

The baseline remains correct while TextCNN fails on **76** examples.

This produces a net gain of:

**129 test examples.**

### TextCNN Fixes by True Class

| True class | Count |
|---|---:|
| anger | 52 |
| love | 52 |
| fear | 45 |
| surprise | 28 |
| sadness | 22 |
| joy | 6 |

### Baseline Confusions Repaired by TextCNN

| True class | Predicted class | Count |
|---|---|---:|
| love | joy | 39 |
| anger | joy | 28 |
| fear | joy | 26 |
| anger | sadness | 19 |
| surprise | joy | 18 |
| fear | sadness | 17 |
| sadness | joy | 16 |
| love | sadness | 11 |
| sadness | anger | 5 |
| surprise | sadness | 5 |
| joy | sadness | 4 |
| anger | fear | 4 |
| surprise | fear | 4 |
| fear | anger | 2 |
| love | anger | 2 |

These cohorts are more informative than aggregate accuracy alone because they identify where the sequence-aware TextCNN changes behavior relative to the lexical TF-IDF baseline.

## TextCNN Regressions

TextCNN introduces **76** errors on examples that the baseline classified correctly.

### Regressions by True Class

| True class | Count |
|---|---:|
| joy | 38 |
| sadness | 19 |
| fear | 11 |
| anger | 5 |
| love | 2 |
| surprise | 1 |

### TextCNN Regression Confusions

| True class | Predicted class | Count |
|---|---|---:|
| joy | love | 30 |
| sadness | joy | 8 |
| sadness | anger | 7 |
| fear | surprise | 7 |
| anger | fear | 4 |
| joy | surprise | 4 |
| sadness | fear | 3 |
| fear | anger | 2 |
| love | joy | 2 |
| joy | sadness | 2 |
| joy | anger | 2 |
| anger | joy | 1 |
| fear | sadness | 1 |
| fear | love | 1 |
| sadness | love | 1 |

These errors prevent the comparison from being reduced to the statement that TextCNN is universally better. The model is better in aggregate but still loses information or makes poorer decisions on some examples.

## Errors Shared by All Models

All three models agree on the same wrong prediction for **93** examples, representing **4.65%** of the test split.

### Shared Errors by True Class

| True class | Count |
|---|---:|
| love | 21 |
| sadness | 17 |
| anger | 16 |
| surprise | 16 |
| fear | 12 |
| joy | 11 |

### Most Common Shared Confusions

| True class | Predicted class | Count |
|---|---|---:|
| love | joy | 20 |
| surprise | fear | 11 |
| anger | sadness | 10 |
| joy | love | 10 |
| sadness | joy | 9 |
| fear | sadness | 8 |
| sadness | anger | 6 |
| anger | joy | 4 |
| surprise | joy | 4 |
| fear | anger | 3 |
| anger | fear | 2 |
| sadness | fear | 2 |
| joy | sadness | 1 |
| fear | surprise | 1 |
| love | anger | 1 |

### Example Shared Errors

| Example | True | Baseline | NumPy MLP | TextCNN | Text |
|---|---|---|---|---|---|
| `dair_test_000010` | fear | anger | anger | anger | i don t feel particularly agitated |
| `dair_test_000067` | anger | sadness | sadness | sadness | i feel a bit stressed even though all the things i have going on are fun |
| `dair_test_000072` | surprise | fear | fear | fear | i am right handed however i play billiards left handed naturally so me trying to play right handed feels weird |
| `dair_test_000103` | fear | anger | anger | anger | i feel agitated with myself that i did not foresee her frustrations earlier leading to the ending of our relationship |
| `dair_test_000131` | love | joy | joy | joy | im feeling generous today heres one more you may have already seen but is good for a chuckle |
| `dair_test_000152` | surprise | fear | fear | fear | i was so uncomfortable and feeling weird feelings but wasn t sure if they were contractions since i never really felt contractions with jared until they jacked me up with pitocin |
| `dair_test_000193` | anger | sadness | sadness | sadness | i really dont like quinn because i feel like she will just end up hurting barney and i hated the lame ted robin storyline |
| `dair_test_000229` | fear | sadness | sadness | sadness | i am truly unfortunate the majority of the time i m usually drained but i obtain it hard to get from bed i really feel restless and others |
| `dair_test_000242` | fear | sadness | sadness | sadness | i see you on the pitchers mound at our little league diamond i feel so anxious for you because it looks so isolated over there |
| `dair_test_000249` | surprise | joy | joy | joy | i will practice meditation if i feel overwhelmed and hopefully become successful in peaceful practice |

These cases are particularly important for the later Error Lab. Agreement across three structurally different models suggests that some examples may involve difficult class boundaries, semantic overlap, weak context, or dataset-label ambiguity.

Those possibilities are hypotheses only. A prediction disagreement with the dataset label is not sufficient evidence that the dataset label itself is incorrect.

## Three-Way Disagreement

Only **21** examples produce three different model predictions.

### Three-Way Disagreements by True Class

| True class | Count |
|---|---:|
| fear | 7 |
| anger | 5 |
| surprise | 5 |
| sadness | 3 |
| love | 1 |

### Example Three-Way Disagreements

| Example | True | Baseline | NumPy MLP | TextCNN | Text |
|---|---|---|---|---|---|
| `dair_test_000006` | anger | sadness | anger | fear | i felt anger when at the end of a telephone call |
| `dair_test_000137` | anger | sadness | anger | fear | i actually feel agitated which led to a terrible day yesterday in which i was unable to concentrate on anything and basically piddled the day away |
| `dair_test_000429` | fear | joy | fear | surprise | i feel like in the last year especially i ve gone from a girl to a woman and despite how hesitant i have always been about getting older next year i will be twenty four i am surprised at how pleased i am to have done so |
| `dair_test_000564` | anger | joy | anger | sadness | i started feeling a little stressed about leaving on time and making sure we got the getting ready pictures i wanted but everything seemed to workout perfectly |
| `dair_test_000910` | sadness | joy | sadness | love | i really wish i had more time to explore twitter as i feel like i lost a lot of time learning how to use the site |
| `dair_test_001006` | fear | sadness | anger | fear | i feel like i have to shy away from triggering some stereotype of a person who will scream and break things because they didnt get to eat their favorite kind of sandwich |
| `dair_test_001150` | sadness | sadness | fear | anger | when i heard the last regulation of the socialist govrenment concerning pensions |
| `dair_test_001202` | surprise | sadness | joy | surprise | i started feeling a little funny but this was not anxiety but at the time i didnt know so i started to tell my brother man i dont feel good and he said whats wrong i said i dont know but u better drive so i pulled over and let him drive |
| `dair_test_001234` | surprise | joy | fear | surprise | i stay the more distanced from others i feel it is strange because i sometimes feel like a new friendship is growing or forming |
| `dair_test_001252` | anger | joy | fear | anger | i am waking up in the middle of the night again with aches and pains and generally feeling grumpy |

These examples provide a compact high-value cohort for manual inspection because each model family interprets the same text differently.

## Main Findings

1. **Model complexity produces measurable gains.**  
   Performance improves from Logistic Regression to the NumPy MLP and again from the NumPy MLP to TextCNN.

2. **TextCNN's improvement is behaviorally observable, not only metric-level.**  
   It repairs 205 baseline errors while introducing 76 regressions.

3. **Most examples are not useful for distinguishing model behavior.**  
   The all-model agreement rate is 82.65%, so the most informative analysis lies in disagreement and shared-error cohorts.

4. **Shared errors remain important despite the stronger neural model.**  
   All three systems agree and fail on 93 examples.

5. **The comparison supports an Error Lab rather than a leaderboard-only presentation.**  
   The meaningful question is not only which model has the highest score, but which language patterns and label boundaries cause models to behave differently.

## Limitations

- The models use different input representations, which is intentional but means internal feature spaces are not directly comparable.
- Raw probability-like scores should not be compared across model families as though they were equally calibrated.
- The analysis is based on one verified held-out dataset split.
- Dataset labels are treated as evaluation targets, not universal psychological truth.
- Some apparent errors may reflect ambiguity or semantic overlap rather than a simple model failure.
- Interpretation of TextCNN internal activations is handled separately and should not be confused with causal explanations of emotion.

## Next Step

The next phase is **P7 — TextCNN Interpretation Lab**.

That phase will inspect convolutional activation regions and map high-activation filters back to exact input token spans, allowing TmotionLab to investigate why particular local text patterns strongly activate the production model.
