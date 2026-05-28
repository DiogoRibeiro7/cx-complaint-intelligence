"""Embedding-based topic discovery, alongside the TF-IDF + NMF baseline.

This module fits an interpretable topic model on top of sentence embeddings:

1. **Encode** each complaint narrative with a Sentence-Transformer.
2. **Cluster** the embeddings with K-means (deterministic, easy to explain).
3. **Label** each cluster with class-based TF-IDF terms &mdash; the standard
   "c-TF-IDF" approach BERTopic popularised. The cluster's documents are
   concatenated, then a global TF-IDF is fit and the top weighted terms per
   pseudo-document become the topic label.

The output mirrors the schema of :class:`cx_complaints.nlp.topics.TopicModelResult`
so downstream code (reports, dashboard, notebooks) can treat the two methods
interchangeably.

Design notes:

- The encoder is injected as a callable so the function is unit-testable
  without downloading a model. Production callers pass
  :class:`SentenceTransformerEncoder`; tests pass a synthetic encoder.
- Sentence-Transformers is imported lazily inside the encoder factory so that
  importing this module is free for users who never instantiate the default
  encoder.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

DEFAULT_ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextEncoder(Protocol):
    """Callable that maps a sequence of strings to an ``(n, d)`` embedding array."""

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        ...


@dataclass
class EmbeddingTopicResult:
    """Artifacts produced by :func:`fit_embedding_topics`.

    The fields mirror :class:`cx_complaints.nlp.topics.TopicModelResult` so the
    two methods can be swapped at the call site.
    """

    encoder_name: str
    embeddings: np.ndarray
    cluster_model: KMeans
    topic_terms: pd.DataFrame
    assignments: pd.DataFrame


class SentenceTransformerEncoder:
    """Default encoder backed by ``sentence-transformers``.

    The Sentence-Transformer library is imported lazily so that this module
    can be imported without pulling torch into memory.
    """

    def __init__(self, model_name: str = DEFAULT_ENCODER_NAME) -> None:
        self.model_name = model_name
        self._model: object | None = None

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode ``texts`` into a dense ``(n, d)`` array."""
        from sentence_transformers import SentenceTransformer

        if self._model is None:
            self._model = SentenceTransformer(self.model_name)

        model = self._model
        encoded = model.encode(  # type: ignore[attr-defined]
            list(texts),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(encoded, dtype=np.float32)


def fit_embedding_topics(
    texts: pd.Series,
    *,
    n_topics: int = 8,
    encoder: TextEncoder | None = None,
    encoder_name: str = DEFAULT_ENCODER_NAME,
    random_state: int = 42,
    top_terms_per_topic: int = 12,
    min_df: int = 2,
    max_features: int = 5_000,
) -> EmbeddingTopicResult:
    """Cluster sentence embeddings and label each cluster with c-TF-IDF terms.

    Parameters
    ----------
    texts:
        Cleaned complaint narratives. Empty rows are silently dropped and not
        encoded, but the returned ``assignments`` frame is aligned to the
        original index of ``texts`` (rows without text get ``topic_id = -1``).
    n_topics:
        Number of clusters to fit.
    encoder:
        Optional encoder. Defaults to :class:`SentenceTransformerEncoder` with
        :data:`DEFAULT_ENCODER_NAME` &mdash; instantiated lazily so tests can
        inject a fake encoder.
    encoder_name:
        Recorded in the result for traceability. Used only when ``encoder`` is
        ``None``.
    random_state:
        K-means seed for reproducibility.
    top_terms_per_topic:
        Number of c-TF-IDF terms to keep per cluster.
    min_df:
        TF-IDF ``min_df`` for the c-TF-IDF labelling step. Lower than the NMF
        baseline because the c-TF-IDF documents are cluster aggregates and
        thus much fewer in number.
    max_features:
        Maximum vocabulary size for c-TF-IDF.
    """
    if n_topics <= 0:
        msg = "n_topics must be positive."
        raise ValueError(msg)
    if top_terms_per_topic <= 0:
        msg = "top_terms_per_topic must be positive."
        raise ValueError(msg)

    cleaned = texts.fillna("").astype(str)
    non_empty_mask = cleaned.str.strip() != ""
    non_empty_texts = cleaned[non_empty_mask]

    if len(non_empty_texts) < n_topics:
        msg = f"Need at least {n_topics} non-empty texts to fit {n_topics} topics."
        raise ValueError(msg)

    active_encoder: TextEncoder = encoder if encoder is not None else SentenceTransformerEncoder(
        encoder_name
    )
    embeddings = active_encoder.encode(non_empty_texts.tolist())
    if embeddings.ndim != 2:
        msg = "Encoder must return a 2D embedding matrix."
        raise ValueError(msg)
    if embeddings.shape[0] != len(non_empty_texts):
        msg = (
            f"Encoder returned {embeddings.shape[0]} embeddings for "
            f"{len(non_empty_texts)} texts."
        )
        raise ValueError(msg)

    cluster_model = KMeans(
        n_clusters=n_topics,
        random_state=random_state,
        n_init=10,
    )
    cluster_assignments = cluster_model.fit_predict(embeddings)

    topic_terms = _extract_cluster_tfidf_terms(
        documents=non_empty_texts,
        cluster_assignments=cluster_assignments,
        n_topics=n_topics,
        top_n=top_terms_per_topic,
        min_df=min_df,
        max_features=max_features,
    )

    # Map distances to similarities so the score column is monotone-positive
    # like the NMF baseline. Smaller distance -> higher score.
    distances_to_centroid = np.linalg.norm(
        embeddings - cluster_model.cluster_centers_[cluster_assignments],
        axis=1,
    )
    similarity = 1.0 / (1.0 + distances_to_centroid)

    assignments = pd.DataFrame(
        {
            "topic_id": np.full(len(texts), -1, dtype=int),
            "topic_score": np.zeros(len(texts), dtype=float),
        },
        index=texts.index,
    )
    assignments.loc[non_empty_mask.values, "topic_id"] = cluster_assignments.astype(int)
    assignments.loc[non_empty_mask.values, "topic_score"] = similarity.astype(float)

    resolved_encoder_name = (
        encoder_name if encoder is None else getattr(encoder, "model_name", encoder_name)
    )

    return EmbeddingTopicResult(
        encoder_name=resolved_encoder_name,
        embeddings=embeddings,
        cluster_model=cluster_model,
        topic_terms=topic_terms,
        assignments=assignments,
    )


def _extract_cluster_tfidf_terms(
    documents: pd.Series,
    cluster_assignments: np.ndarray,
    n_topics: int,
    *,
    top_n: int,
    min_df: int,
    max_features: int,
) -> pd.DataFrame:
    """Build the c-TF-IDF topic-terms table.

    Each cluster contributes one "pseudo-document" (its concatenated texts).
    A TF-IDF model is fit over those pseudo-documents and the highest-weighted
    terms per row become the topic label. This is the same labelling trick
    BERTopic uses.
    """
    cluster_documents: list[str] = []
    for cluster_id in range(n_topics):
        mask = cluster_assignments == cluster_id
        cluster_texts = documents.iloc[mask].tolist()
        cluster_documents.append(" ".join(cluster_texts) if cluster_texts else "")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(cluster_documents)
    if not sparse.issparse(matrix) or matrix.shape[1] == 0:
        msg = (
            "Cluster-level TF-IDF produced an empty vocabulary. "
            "Lower min_df or provide more text."
        )
        raise ValueError(msg)

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    dense_weights = matrix.toarray()

    rows: list[dict[str, object]] = []
    for cluster_id in range(n_topics):
        weights = dense_weights[cluster_id]
        if weights.sum() == 0:
            continue
        top_indices = np.argsort(weights)[::-1][:top_n]
        for rank, term_idx in enumerate(top_indices, start=1):
            if weights[term_idx] <= 0:
                break
            rows.append(
                {
                    "topic_id": cluster_id,
                    "rank": rank,
                    "term": feature_names[term_idx],
                    "weight": float(weights[term_idx]),
                }
            )

    return pd.DataFrame(rows, columns=["topic_id", "rank", "term", "weight"])
