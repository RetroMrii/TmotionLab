# TmotionLab Error Lab Analysis

## Purpose

This report analyzes TextCNN predictions that disagree with the DAIR.AI Emotion test labels.

The analysis is benchmark-relative: a row is treated as an evaluation error when the model prediction differs from the dataset label.

**Important caveat:** this does not imply that every disagreement is a clear semantic model failure. Some examples may reflect ambiguous emotion boundaries, mixed emotions, limited context, or annotation noise. No systematic manual adjudication or dataset relabeling was performed in this version.

## Overall Error Summary

- **Model:** `textcnn_dair_v001`
- **Test examples:** 2000
- **Benchmark disagreements:** 207
- **Benchmark error rate:** 0.1035
- **High-score threshold:** 0.80
- **High-score disagreements:** 104
- **Unique confusion pairs:** 22
- **Interpretation available:** 20

## Dominant Confusion Pairs

| True label | Predicted label | Count | Mean score | Max score | High-score disagreements |
|---|---|---:|---:|---:|---:|
| joy | love | 40 | 0.8120 | 0.9989 | 24 |
| love | joy | 28 | 0.8112 | 0.9945 | 17 |
| sadness | joy | 24 | 0.6488 | 0.9149 | 7 |
| sadness | anger | 15 | 0.8104 | 0.9999 | 10 |
| surprise | fear | 13 | 0.8032 | 0.9947 | 7 |
| anger | sadness | 11 | 0.8043 | 0.9841 | 7 |
| fear | sadness | 11 | 0.8041 | 0.9936 | 7 |
| fear | surprise | 10 | 0.7273 | 0.9723 | 4 |
| anger | fear | 9 | 0.6123 | 0.9511 | 2 |
| fear | anger | 7 | 0.8305 | 0.9692 | 5 |
| surprise | joy | 7 | 0.7457 | 0.9347 | 3 |
| anger | joy | 7 | 0.7069 | 0.9784 | 3 |
| sadness | fear | 6 | 0.7659 | 0.9933 | 2 |
| joy | surprise | 4 | 0.8266 | 0.9670 | 2 |
| joy | sadness | 4 | 0.7588 | 0.9932 | 2 |

These counts describe disagreement with the benchmark label. They do not establish that the model prediction is semantically unreasonable.

## High-Score Benchmark Disagreements

These are disagreements where the TextCNN softmax score is at least 0.80.

The score is not treated as calibrated confidence.

| Example | True | TextCNN | Score | Baseline | NumPy | Text |
|---|---|---|---:|---|---|---|
| `dair_test_001837` | sadness | anger | 0.9999 | anger | anger | i hate these feelings in my heart i hate that work stressed me out i hate that cornelius wont let me get my way im frustrated lord |
| `dair_test_001698` | sadness | anger | 0.9997 | sadness | anger | i work well with almost every client ive ever been in contact with because i know what it means to feel depressed angry frustrated irritated hopeless and apathetic because i feel it daily |
| `dair_test_001943` | sadness | anger | 0.9996 | anger | anger | i didn t mean to get angry with you bommie i just can t control my feelings hellip i just hated myself why i am like this the dara who can t get over with that b |
| `dair_test_000353` | joy | love | 0.9989 | love | love | i could feel his breath on me and smell the sweet scent of him |
| `dair_test_001537` | joy | love | 0.9953 | joy | love | i like the fresh feeling of sweet he gave me |
| `dair_test_001275` | surprise | fear | 0.9947 | fear | fear | i feel so weird and scattered with all wonders about a million different things |
| `dair_test_000152` | surprise | fear | 0.9946 | fear | fear | i was so uncomfortable and feeling weird feelings but wasn t sure if they were contractions since i never really felt contractions with jared until they jacked me up with pitocin |
| `dair_test_000265` | love | joy | 0.9945 | joy | joy | i feel so blessed and honored that we get to be its parents |
| `dair_test_000869` | love | joy | 0.9944 | joy | joy | i feel like if people accepted that wed get along a lot better |
| `dair_test_001677` | fear | sadness | 0.9936 | joy | fear | i become someone else and i make random awkward jokes honestly this feeling is so strange is this what it feels like to be on top of a cloud |
| `dair_test_001026` | sadness | fear | 0.9933 | fear | fear | i have not conducted a survey but it is quite likely that many of them feel as assaulted by onel s demons and other creators as i would have felt had the walls been covered only with eminent figures patriotic heroes and epic deeds |
| `dair_test_001786` | joy | sadness | 0.9932 | sadness | joy | im feeling all jolly and warm inside but i just feel empty |

## Shared Disagreements Across All Models

All three model families produce the same wrong benchmark prediction on **93** examples.

These examples are especially useful because Logistic Regression, the NumPy MLP, and TextCNN arrive at the same alternative label despite using different representations and architectures.

Cross-model agreement is evidence of a stable pattern, not proof that the benchmark annotation is wrong.

| Example | True | TextCNN | Score | Baseline | NumPy | Text |
|---|---|---|---:|---|---|---|
| `dair_test_001837` | sadness | anger | 0.9999 | anger | anger | i hate these feelings in my heart i hate that work stressed me out i hate that cornelius wont let me get my way im frustrated lord |
| `dair_test_001943` | sadness | anger | 0.9996 | anger | anger | i didn t mean to get angry with you bommie i just can t control my feelings hellip i just hated myself why i am like this the dara who can t get over with that b |
| `dair_test_000353` | joy | love | 0.9989 | love | love | i could feel his breath on me and smell the sweet scent of him |
| `dair_test_001275` | surprise | fear | 0.9947 | fear | fear | i feel so weird and scattered with all wonders about a million different things |
| `dair_test_000152` | surprise | fear | 0.9946 | fear | fear | i was so uncomfortable and feeling weird feelings but wasn t sure if they were contractions since i never really felt contractions with jared until they jacked me up with pitocin |
| `dair_test_000265` | love | joy | 0.9945 | joy | joy | i feel so blessed and honored that we get to be its parents |
| `dair_test_000869` | love | joy | 0.9944 | joy | joy | i feel like if people accepted that wed get along a lot better |
| `dair_test_001026` | sadness | fear | 0.9933 | fear | fear | i have not conducted a survey but it is quite likely that many of them feel as assaulted by onel s demons and other creators as i would have felt had the walls been covered only with eminent figures patriotic heroes and epic deeds |
| `dair_test_000067` | anger | sadness | 0.9841 | sadness | sadness | i feel a bit stressed even though all the things i have going on are fun |
| `dair_test_001887` | joy | love | 0.9792 | love | love | i feel like some of you have pains and you cannot imagine becoming passionate about the group or the idea that is causing pain |
| `dair_test_001876` | anger | sadness | 0.9786 | sadness | sadness | i need a break or im feeling stressed out |
| `dair_test_000816` | anger | joy | 0.9784 | joy | joy | whenever i put myself in others shoes and try to make the person happy |

## TextCNN Regressions Relative to Baseline

The Logistic Regression baseline is benchmark-correct while TextCNN is benchmark-wrong on **76** examples.

These cases are useful because they show where the more complex sequence model performs worse than the simpler lexical baseline.

| Example | True | TextCNN | Score | Baseline | NumPy | Text |
|---|---|---|---:|---|---|---|
| `dair_test_001698` | sadness | anger | 0.9997 | sadness | anger | i work well with almost every client ive ever been in contact with because i know what it means to feel depressed angry frustrated irritated hopeless and apathetic because i feel it daily |
| `dair_test_001537` | joy | love | 0.9953 | joy | love | i like the fresh feeling of sweet he gave me |
| `dair_test_000413` | joy | love | 0.9929 | joy | joy | i am and always have been a very sincere nice feeling sociable compassionate helpful girl |
| `dair_test_001739` | love | joy | 0.9902 | love | love | im feeling all romantic so i thought id show you some easy last minute presents and fashion ideas for valentines day |
| `dair_test_000997` | joy | love | 0.9840 | joy | love | i chant the invocation and feel his force supporting me as i teach |
| `dair_test_000720` | joy | love | 0.9807 | joy | love | i think your viewers tonight will enjoy the show coming from malm they will like some things be less fond of other things but hopefully they will feel entertained and smitten and feel the urge to cast a vote regardless if your country is voting tonight or not |
| `dair_test_001607` | joy | love | 0.9784 | joy | love | i felt it had a slight bitterness in the finish that detracted from its oily mouthfeel and sweet entry |
| `dair_test_000585` | fear | sadness | 0.9728 | fear | fear | i grabbed my dog and hugged her fiercly for the next hour or so until i began to feel a bit like myself again but i havent completly shaken the feeling and have been feeling rather depressed anxious all day |
| `dair_test_001322` | fear | surprise | 0.9696 | fear | fear | i started to feel so overwhelmed |
| `dair_test_000415` | joy | surprise | 0.9670 | joy | joy | i loved the feeling i got during an amazing slalom run whether it was in training or in a race |
| `dair_test_001206` | joy | love | 0.9616 | joy | joy | ive come to feel about a supporting character in one of my all time favorite films giant |
| `dair_test_000306` | anger | fear | 0.9511 | anger | fear | i took for granted a few weeks ago is really weird and makes me feel really agitated and frustrated |

## Three-Way Disagreements

TextCNN, NumPy MLP, and Logistic Regression all choose different labels on **9** TextCNN benchmark-error examples.

| Example | True | TextCNN | Score | Baseline | NumPy | Text |
|---|---|---|---:|---|---|---|
| `dair_test_001677` | fear | sadness | 0.9936 | joy | fear | i become someone else and i make random awkward jokes honestly this feeling is so strange is this what it feels like to be on top of a cloud |
| `dair_test_000429` | fear | surprise | 0.9723 | joy | fear | i feel like in the last year especially i ve gone from a girl to a woman and despite how hesitant i have always been about getting older next year i will be twenty four i am surprised at how pleased i am to have done so |
| `dair_test_000137` | anger | fear | 0.7868 | sadness | anger | i actually feel agitated which led to a terrible day yesterday in which i was unable to concentrate on anything and basically piddled the day away |
| `dair_test_001560` | sadness | joy | 0.7244 | sadness | fear | i feel helpless to regain a safe feeling |
| `dair_test_000564` | anger | sadness | 0.5870 | joy | anger | i started feeling a little stressed about leaving on time and making sure we got the getting ready pictures i wanted but everything seemed to workout perfectly |
| `dair_test_001911` | surprise | fear | 0.5263 | joy | surprise | im ok with that it feels a little weird |
| `dair_test_001150` | sadness | anger | 0.4542 | sadness | fear | when i heard the last regulation of the socialist govrenment concerning pensions |
| `dair_test_000910` | sadness | love | 0.4401 | joy | sadness | i really wish i had more time to explore twitter as i feel like i lost a lot of time learning how to use the site |
| `dair_test_000006` | anger | fear | 0.3968 | sadness | anger | i felt anger when at the end of a telephone call |

## Interpretation Coverage

Interpretation metadata is available for **20** of the 207 benchmark disagreements.

This coverage comes from the P7 held-out interpretation sample and is intentionally partial.

| Example | True | TextCNN | Score | Baseline | NumPy | Text |
|---|---|---|---:|---|---|---|
| `dair_test_001837` | sadness | anger | 0.9999 | anger | anger | i hate these feelings in my heart i hate that work stressed me out i hate that cornelius wont let me get my way im frustrated lord |
| `dair_test_001698` | sadness | anger | 0.9997 | sadness | anger | i work well with almost every client ive ever been in contact with because i know what it means to feel depressed angry frustrated irritated hopeless and apathetic because i feel it daily |
| `dair_test_001943` | sadness | anger | 0.9996 | anger | anger | i didn t mean to get angry with you bommie i just can t control my feelings hellip i just hated myself why i am like this the dara who can t get over with that b |
| `dair_test_000353` | joy | love | 0.9989 | love | love | i could feel his breath on me and smell the sweet scent of him |
| `dair_test_001537` | joy | love | 0.9953 | joy | love | i like the fresh feeling of sweet he gave me |
| `dair_test_001275` | surprise | fear | 0.9947 | fear | fear | i feel so weird and scattered with all wonders about a million different things |
| `dair_test_000152` | surprise | fear | 0.9946 | fear | fear | i was so uncomfortable and feeling weird feelings but wasn t sure if they were contractions since i never really felt contractions with jared until they jacked me up with pitocin |
| `dair_test_000265` | love | joy | 0.9945 | joy | joy | i feel so blessed and honored that we get to be its parents |
| `dair_test_000869` | love | joy | 0.9944 | joy | joy | i feel like if people accepted that wed get along a lot better |
| `dair_test_001677` | fear | sadness | 0.9936 | joy | fear | i become someone else and i make random awkward jokes honestly this feeling is so strange is this what it feels like to be on top of a cloud |

## Error Lab Interpretation Rules

The current Error Lab uses the following terminology:

- **Benchmark disagreement:** model prediction differs from the dataset label.
- **Evaluation error:** equivalent benchmark-relative term used for metric analysis.
- **High-score disagreement:** benchmark disagreement with a high raw softmax score.

The following claims are deliberately avoided without human adjudication:

- the model is definitely semantically wrong;
- the dataset annotation is definitely wrong;
- a high score proves model confidence is calibrated;
- all shared-model disagreements are mislabeled data;

## Deferred Analysis

The following extensions are intentionally deferred:

- systematic manual review of all disagreements;
- analyst-authored error taxonomy;
- dataset relabeling or annotation-quality study;
- multi-reviewer agreement analysis;
- calibration analysis;

## Main Takeaway

The TextCNN achieves strong benchmark performance, but its remaining disagreements are not uniform.

Some are model-specific regressions, some are shared across all three architectures, and some occur with very high output scores.

TmotionLab therefore treats error analysis as inspection of model behavior relative to the benchmark rather than assuming that every label disagreement has a single obvious cause.