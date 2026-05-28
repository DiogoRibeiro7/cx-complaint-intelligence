from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from cx_complaints.nlp.embedding_topics import (
    DEFAULT_ENCODER_NAME,
    EmbeddingTopicResult,
    fit_embedding_topics,
)


class _DeterministicEncoder:
    """Fake encoder that places texts on three well-separated clusters.

    The first cluster's texts contain ``alpha``, the second ``beta``, and the
    third ``gamma``. The encoder returns a 4-dimensional vector that puts each
    text near a corner of the simplex, which makes K-means recovery trivial
    and deterministic.
    """

    model_name = "test-fake-encoder"

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "alpha" in lowered:
                vectors.append([1.0, 0.0, 0.0, 0.0])
            elif "beta" in lowered:
                vectors.append([0.0, 1.0, 0.0, 0.0])
            elif "gamma" in lowered:
                vectors.append([0.0, 0.0, 1.0, 0.0])
            else:
                vectors.append([0.25, 0.25, 0.25, 0.25])
        return np.asarray(vectors, dtype=np.float32)


def _sample_texts() -> pd.Series:
    return pd.Series(
        [
            "alpha account overdraft fee",
            "alpha overdraft charged twice",
            "alpha account closed without notice",
            "beta credit report incorrect information",
            "beta credit dispute investigation",
            "beta report inaccurate information",
            "gamma debt collection harassment",
            "gamma collector calls late at night",
            "gamma debt not owed harassment",
        ],
        index=range(9),
    )


def test_fit_embedding_topics_returns_result_with_expected_shape() -> None:
    result = fit_embedding_topics(
        _sample_texts(),
        n_topics=3,
        encoder=_DeterministicEncoder(),
        min_df=1,
    )

    assert isinstance(result, EmbeddingTopicResult)
    assert result.embeddings.shape == (9, 4)
    assert result.encoder_name == "test-fake-encoder"
    assert set(result.assignments.columns) == {"topic_id", "topic_score"}
    assert len(result.assignments) == 9
    assert (result.assignments["topic_id"] >= 0).all()


def test_fit_embedding_topics_groups_by_keyword() -> None:
    texts = _sample_texts()
    result = fit_embedding_topics(
        texts,
        n_topics=3,
        encoder=_DeterministicEncoder(),
        min_df=1,
    )

    keywords = ["alpha"] * 3 + ["beta"] * 3 + ["gamma"] * 3
    grouping = (
        pd.DataFrame({"keyword": keywords})
        .assign(topic=result.assignments["topic_id"].values)
        .groupby("keyword")["topic"]
        .nunique()
    )
    assert (grouping == 1).all(), "Each keyword group should land in exactly one cluster."


def test_fit_embedding_topics_topic_terms_have_correct_schema() -> None:
    result = fit_embedding_topics(
        _sample_texts(),
        n_topics=3,
        encoder=_DeterministicEncoder(),
        top_terms_per_topic=5,
        min_df=1,
    )

    assert list(result.topic_terms.columns) == ["topic_id", "rank", "term", "weight"]
    assert set(result.topic_terms["topic_id"].unique()) == {0, 1, 2}
    # Each cluster's labels should mention one of the seed keywords.
    for cluster_id in range(3):
        terms = (
            result.topic_terms.loc[result.topic_terms["topic_id"] == cluster_id, "term"]
            .tolist()
        )
        assert any(seed in " ".join(terms) for seed in ("alpha", "beta", "gamma"))


def test_fit_embedding_topics_assigns_empty_texts_to_minus_one() -> None:
    texts = pd.concat([_sample_texts(), pd.Series(["", None], index=[10, 11])])
    result = fit_embedding_topics(
        texts,
        n_topics=3,
        encoder=_DeterministicEncoder(),
        min_df=1,
    )

    assert int(result.assignments.loc[10, "topic_id"]) == -1
    assert int(result.assignments.loc[11, "topic_id"]) == -1
    assert float(result.assignments.loc[10, "topic_score"]) == 0.0


def test_fit_embedding_topics_rejects_too_few_texts() -> None:
    texts = pd.Series(["only one"])
    with pytest.raises(ValueError, match="at least 3 non-empty"):
        fit_embedding_topics(
            texts,
            n_topics=3,
            encoder=_DeterministicEncoder(),
            min_df=1,
        )


def test_fit_embedding_topics_rejects_invalid_n_topics() -> None:
    with pytest.raises(ValueError, match="n_topics must be positive"):
        fit_embedding_topics(
            _sample_texts(),
            n_topics=0,
            encoder=_DeterministicEncoder(),
        )


def test_default_encoder_name_constant_matches_expected_model() -> None:
    assert DEFAULT_ENCODER_NAME == "sentence-transformers/all-MiniLM-L6-v2"
