import numpy as np

from ml.src.models.baseline_tfidf import (
    build_baseline_pipeline,
)


def test_baseline_fit_predict_and_probability_shape():
    texts = [
        "i feel very happy",
        "this makes me joyful",
        "i am extremely sad",
        "this is deeply upsetting",
        "i feel happy today",
        "i feel sad today",
    ]

    labels = [
        1,
        1,
        0,
        0,
        1,
        0,
    ]

    vectorizer_config = {
        "lowercase": True,
        "ngram_range": [1, 2],
        "min_df": 1,
        "max_df": 1.0,
        "max_features": 100,
        "sublinear_tf": True,
    }

    model_config = {
        "C": 1.0,
        "max_iter": 200,
        "solver": "lbfgs",
        "class_weight": None,
    }

    model = build_baseline_pipeline(
        vectorizer_config,
        model_config,
        seed=42,
    )

    model.fit(
        texts,
        labels,
    )

    predictions = model.predict(texts)
    probabilities = model.predict_proba(texts)

    assert predictions.shape == (6,)
    assert probabilities.shape == (6, 2)

    np.testing.assert_allclose(
        probabilities.sum(axis=1),
        np.ones(6),
        atol=1e-7,
    )


def test_baseline_is_deterministic():
    texts = [
        "happy joyful",
        "joy happy",
        "sad upset",
        "sad unhappy",
    ]

    labels = [1, 1, 0, 0]

    vectorizer_config = {
        "lowercase": True,
        "ngram_range": [1, 1],
        "min_df": 1,
        "max_df": 1.0,
        "max_features": 100,
        "sublinear_tf": True,
    }

    model_config = {
        "C": 1.0,
        "max_iter": 200,
        "solver": "lbfgs",
        "class_weight": None,
    }

    first = build_baseline_pipeline(
        vectorizer_config,
        model_config,
        seed=42,
    )

    second = build_baseline_pipeline(
        vectorizer_config,
        model_config,
        seed=42,
    )

    first.fit(texts, labels)
    second.fit(texts, labels)

    assert np.array_equal(
        first.predict(texts),
        second.predict(texts),
    )
