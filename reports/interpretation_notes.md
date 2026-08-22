# TmotionLab TextCNN Interpretation Notes

## Purpose

This report documents the first validated interpretation layer for `textcnn_dair_v001`.

The interpretation system exposes high-activation convolutional regions from the trained TextCNN and maps their activation positions back to exact token spans in the input text.

These outputs are **model-internal evidence**. They are not causal explanations of a person's emotion and should not be interpreted as psychological ground truth.

## Interpretation Method

The TextCNN architecture processes token sequences through:

```text
token IDs
-> embedding layer
-> Conv1D filters with kernel sizes 3, 4, and 5
-> ReLU
-> global max pooling
-> concatenation
-> classifier
```

For each convolutional filter, the interpretation pipeline records the position with the largest post-ReLU activation.

The corresponding token window is reconstructed using:

- kernel size;
- filter index;
- activation value;
- start token index;
- end token index;
- exact token span.

The model itself is not retrained or modified for interpretation.

## Validation Sample

The interpretation validation artifact contains **40** held-out examples.

Cohorts:

- correct, high-score predictions;
- correct, low-score predictions;
- wrong, high-score predictions;
- wrong, low-score predictions.

Observed totals:

- **Correct examples:** 20
- **Wrong examples:** 20

The sample is deliberately balanced between correct and incorrect predictions so that interpretation behavior is inspected on failures as well as successes.

## Padding-Overlap Behavior

A convolutional filter can reach its raw maximum on a window that overlaps right-padding.

This matters because the TextCNN uses global max pooling and convolutional bias terms. A high activation is therefore not automatically guaranteed to correspond entirely to real text.

Across the interpretation sample:

- **Mean filters with padding-overlap maxima:** 87.58
- **Median:** 75.00
- **Minimum:** 0
- **Maximum:** 286

User-visible interpretation spans exclude windows that overlap padding, but the raw padding-overlap count is preserved as diagnostic metadata.

### Padding Behavior by Cohort

| Cohort | Mean padding maxima | Median | Max |
|---|---:|---:|---:|
| correct_high_score | 146.20 | 150.50 | 286 |
| correct_low_score | 68.70 | 64.50 | 173 |
| wrong_high_score | 49.20 | 48.00 | 79 |
| wrong_low_score | 86.20 | 74.50 | 140 |

A higher padding-overlap count in one cohort does not establish that padding causes prediction errors. It only identifies a behavior worth further investigation.

## Frequently Observed Filters

The following filters appear most frequently among the top valid activation spans in the sampled examples:

| Kernel | Filter | Predicted class | Unique examples | Mean activation | Mean rank |
|---:|---:|---|---:|---:|---:|
| 3 | 120 | sadness | 5 | 4.6488 | 3.60 |
| 4 | 67 | sadness | 5 | 4.1443 | 6.80 |
| 4 | 112 | fear | 5 | 3.4400 | 3.20 |
| 4 | 97 | sadness | 4 | 4.9190 | 3.00 |
| 3 | 16 | sadness | 4 | 4.5993 | 3.25 |
| 3 | 22 | sadness | 4 | 4.5207 | 4.75 |
| 3 | 83 | sadness | 4 | 4.2502 | 5.75 |
| 3 | 84 | sadness | 4 | 3.6556 | 6.25 |
| 4 | 29 | anger | 4 | 3.6421 | 3.50 |
| 3 | 51 | joy | 4 | 3.5874 | 5.00 |
| 4 | 91 | joy | 4 | 3.5685 | 5.50 |
| 5 | 17 | sadness | 3 | 5.3163 | 2.00 |
| 4 | 22 | sadness | 3 | 4.3198 | 8.00 |
| 3 | 63 | fear | 3 | 4.0273 | 3.33 |
| 5 | 113 | anger | 3 | 3.7870 | 3.00 |

These observations do **not** justify naming a filter as a permanent emotion detector.

For example, if a filter frequently appears among examples predicted as `fear`, the supported claim is:

> The filter frequently appears among the strongest activation spans for sampled examples predicted as fear.

The unsupported claim would be:

> This is a fear filter.

Class specificity would require broader controlled analysis across the full held-out set.

## Frequently Observed Spans

The most repeated normalized text spans among the sampled top activations are:

| Span | Predicted class | Unique examples | Mean activation |
|---|---|---:|---:|
| `just feel troubled` | sadness | 1 | 3.5173 |
| `school feeling miserable` | sadness | 1 | 3.6976 |
| `do feel superior` | joy | 1 | 3.1937 |
| `feeling of sweet` | love | 1 | 3.6767 |
| `feel dumb and` | sadness | 1 | 4.3429 |
| `feel so exhausted by` | sadness | 1 | 4.3421 |
| `feeling it will not be` | love | 1 | 3.0359 |
| `felt anger when at` | fear | 1 | 2.2831 |
| `feel humiliated embarrassed` | sadness | 1 | 5.2754 |
| `feel humiliated embarrassed or foolish` | sadness | 1 | 4.9672 |
| `i feel ecstatic and` | joy | 1 | 4.3294 |
| `fresh feeling of sweet` | love | 1 | 4.0122 |
| `feel totally useless today` | sadness | 1 | 3.8379 |
| `feel ecstatic and` | joy | 1 | 3.8053 |
| `feel a little stunned but` | love | 1 | 3.7249 |
| `really do feel superior` | joy | 1 | 3.6305 |
| `smell the sweet` | love | 1 | 3.6147 |
| `feel so blessed and` | joy | 1 | 3.5743 |
| `feel so blessed` | joy | 1 | 3.5017 |
| `of school feeling miserable` | sadness | 1 | 2.8699 |

Repeated generic patterns such as `i feel`, `i am`, or similar dataset-specific constructions should be treated cautiously. Their presence may indicate that filters learn recurring dataset syntax rather than purely emotion-specific semantic cues.

## Representative Examples

### Correct High-Score Prediction

- **Example ID:** `dair_test_000527`
- **True label:** sadness
- **Prediction:** sadness
- **Prediction score:** 1.0000
- **Padding-overlap filter maxima:** 53

**Text**

> i made that make me feel dumb and dumber

**Top valid activation spans**

| Rank | Kernel | Filter | Activation | Span |
|---:|---:|---:|---:|---|
| 1 | 3 | 120 | 5.1442 | `me feel dumb` |
| 2 | 3 | 16 | 4.6028 | `me feel dumb` |
| 3 | 3 | 29 | 4.5059 | `feel dumb and` |
| 4 | 3 | 83 | 4.3840 | `feel dumb and` |
| 5 | 4 | 3 | 4.3106 | `me feel dumb and` |

### Correct Lower-Score Prediction

- **Example ID:** `dair_test_000094`
- **True label:** anger
- **Prediction:** anger
- **Prediction score:** 0.3415
- **Padding-overlap filter maxima:** 173

**Text**

> when a friend dropped a frog down my neck

**Top valid activation spans**

| Rank | Kernel | Filter | Activation | Span |
|---:|---:|---:|---:|---|
| 1 | 5 | 73 | 2.5709 | `a friend dropped a frog` |
| 2 | 4 | 64 | 2.5587 | `when a friend dropped` |
| 3 | 3 | 46 | 2.5391 | `a friend dropped` |
| 4 | 5 | 13 | 2.5242 | `dropped a frog down my` |
| 5 | 5 | 110 | 2.3097 | `when a friend dropped a` |

### Wrong High-Score Prediction

- **Example ID:** `dair_test_001837`
- **True label:** sadness
- **Prediction:** anger
- **Prediction score:** 0.9999
- **Padding-overlap filter maxima:** 79

**Text**

> i hate these feelings in my heart i hate that work stressed me out i hate that cornelius wont let me get my way im frustrated lord

**Top valid activation spans**

| Rank | Kernel | Filter | Activation | Span |
|---:|---:|---:|---:|---|
| 1 | 3 | 42 | 3.9282 | `way im frustrated` |
| 2 | 3 | 38 | 3.5917 | `my heart i` |
| 3 | 4 | 80 | 3.5729 | `way im frustrated lord` |
| 4 | 3 | 95 | 3.4033 | `im frustrated lord` |
| 5 | 4 | 111 | 3.3720 | `way im frustrated lord` |

### Wrong Lower-Score Prediction

- **Example ID:** `dair_test_001289`
- **True label:** anger
- **Prediction:** fear
- **Prediction score:** 0.2552
- **Padding-overlap filter maxima:** 71

**Text**

> at a party i met a girl who drew me to her

**Top valid activation spans**

| Rank | Kernel | Filter | Activation | Span |
|---:|---:|---:|---:|---|
| 1 | 5 | 13 | 2.2566 | `met a girl who drew` |
| 2 | 3 | 118 | 2.2302 | `party i met` |
| 3 | 5 | 19 | 2.0970 | `a girl who drew me` |
| 4 | 5 | 0 | 2.0949 | `met a girl who drew` |
| 5 | 5 | 71 | 1.9091 | `girl who drew me to` |

## Interpretation of Correct Predictions

Correct predictions can show whether highly activated local regions are semantically consistent with the model's selected class.

However, semantic consistency is evaluated after observing the model output. It does not prove that a highlighted phrase alone determined the classification.

The strongest evidence supported by this pipeline is:

> These local token regions produced strong internal convolutional responses in the trained TextCNN.

## Interpretation of Incorrect Predictions

Incorrect examples are essential to interpretation validation.

A strongly activated phrase accompanying a wrong prediction demonstrates that:

- high activation does not imply a correct prediction;
- locally informative text may still support the wrong class;
- multiple emotional cues may compete;
- the classifier layer may combine filter activations imperfectly;
- dataset label boundaries may overlap semantically.

Wrong high-score examples are especially important because they expose cases where the model is both confident and incorrect.

## What the Interpretation Supports

The current system supports the following claims:

1. Exact high-activation convolutional windows can be traced back to input token spans.
2. Different filters respond strongly to different local n-gram regions.
3. Activation evidence can be inspected for both correct and incorrect predictions.
4. Padding-overlap maxima occur and must be handled explicitly.
5. Repeated filter/span behavior can be measured reproducibly from saved artifacts.

## What the Interpretation Does Not Support

The current system does **not** support claims that:

- highlighted text caused the predicted emotion;
- activation magnitude is equivalent to class importance;
- a particular filter permanently represents one emotion;
- the model has identified the writer's actual psychological state;
- high activation implies a prediction is correct;
- removing a highlighted phrase would necessarily change the prediction in a predictable way.

Those questions would require additional techniques such as controlled input perturbation or class-specific attribution analysis.

## Current Interpretation Limitations

- The aggregate validation sample contains only 40 held-out examples.
- The sample is cohort-balanced rather than representative of the natural test distribution.
- The method observes post-ReLU convolutional activations before the final classifier.
- Filter activation strength is not directly equivalent to contribution to a particular output logit.
- Global max pooling discards non-maximal activations.
- Padding-overlap behavior exists because the original production model was not trained with explicit convolution-window masking.
- Token-level spans are based on the project's simple tokenizer rather than subword tokenization.
- Interpretation results are specific to `textcnn_dair_v001`.

## Portfolio Significance

The interpretation layer changes TmotionLab from a system that only reports:

```text
text -> emotion prediction
```

into one that can also inspect:

```text
text
-> prediction
-> internal convolutional responses
-> exact activated token regions
-> correct/failure comparison
```

This supports the broader TmotionLab thesis that model behavior should be inspectable rather than reduced to a leaderboard score.

## Next Step

The next phase is **P8 — Error Lab**.

The Error Lab will combine:

- model predictions;
- confusion pairs;
- cross-model comparison cohorts;
- high-score mistakes;
- interpretation evidence;
- manually reviewed error categories.

Interpretation metadata may support error analysis, but it should remain separate from analyst-authored hypotheses about why a prediction failed.