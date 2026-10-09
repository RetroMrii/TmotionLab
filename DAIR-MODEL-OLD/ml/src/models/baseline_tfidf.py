from __future__ import annotations

from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_baseline_pipeline(
    vectorizer_config: dict[str, Any],
    model_config: dict[str, Any],
    *,
    seed: int,
) -> Pipeline:
    ngram_range = tuple(vectorizer_config["ngram_range"])

    vectorizer = TfidfVectorizer(
        lowercase=bool(vectorizer_config["lowercase"]),
        ngram_range=ngram_range,
        min_df=vectorizer_config["min_df"],
        max_df=vectorizer_config["max_df"],
        max_features=vectorizer_config["max_features"],
        sublinear_tf=bool(vectorizer_config["sublinear_tf"]),
    )

    classifier = LogisticRegression(
        C=float(model_config["C"]),
        max_iter=int(model_config["max_iter"]),
        solver=str(model_config["solver"]),
        class_weight=model_config.get("class_weight"),
        random_state=seed,
    )

    return Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier),
        ]
    )
