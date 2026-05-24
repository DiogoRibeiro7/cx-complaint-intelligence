"""Topic discovery using TF-IDF and Non-negative Matrix Factorization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class TopicModelResult:
    """Container with fitted topic model artifacts and tabular outputs."""

    vectorizer: TfidfVectorizer
    model: NMF
    document_topic_matrix: np.ndarray
    topic_terms: pd.DataFrame
    assignments: pd.DataFrame


def fit_nmf_topics(
    texts: pd.Series,
    n_topics: int = 8,
    max_features: int = 5_000,
    min_df: int = 5,
    max_df: float = 0.85,
    random_state: int = 42,
) -> TopicModelResult:
    """Fit a TF-IDF + NMF topic model to complaint narratives.

    Parameters
    ----------
    texts:
        Cleaned complaint narratives.
    n_topics:
        Number of latent topics to discover.
    max_features:
        Maximum vocabulary size for TF-IDF.
    min_df:
        Ignore terms that appear in fewer documents than this value.
    max_df:
        Ignore terms that appear in more than this share of documents.
    random_state:
        Random seed for reproducibility.
    """
    if n_topics <= 0:
        msg = "n_topics must be positive."
        raise ValueError(msg)

    clean_texts = texts.fillna("").astype(str)
    non_empty_count = int((clean_texts.str.strip() != "").sum())
    if non_empty_count < n_topics:
        msg = f"Need at least {n_topics} non-empty texts to fit {n_topics} topics."
        raise ValueError(msg)

    # Keep the baseline transparent and interview-friendly. NMF over TF-IDF gives
    # readable topics and scales well enough for a portfolio project.
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
    )
    tfidf = vectorizer.fit_transform(clean_texts)

    if not sparse.issparse(tfidf) or tfidf.shape[1] == 0:
        msg = "TF-IDF matrix is empty. Try lowering min_df or using more data."
        raise ValueError(msg)

    model = NMF(
        n_components=n_topics,
        init="nndsvda",
        random_state=random_state,
        max_iter=500,
    )
    document_topic_matrix = model.fit_transform(tfidf)
    topic_terms = extract_topic_terms(model, vectorizer, top_n=12)
    assignments = assign_topics(document_topic_matrix)

    return TopicModelResult(
        vectorizer=vectorizer,
        model=model,
        document_topic_matrix=document_topic_matrix,
        topic_terms=topic_terms,
        assignments=assignments,
    )


def extract_topic_terms(model: NMF, vectorizer: TfidfVectorizer, top_n: int = 12) -> pd.DataFrame:
    """Extract the highest-weighted terms for each topic."""
    if top_n <= 0:
        msg = "top_n must be positive."
        raise ValueError(msg)

    terms = np.asarray(vectorizer.get_feature_names_out())
    rows: list[dict[str, object]] = []

    for topic_idx, component in enumerate(model.components_):
        top_indices = np.argsort(component)[::-1][:top_n]
        for rank, term_idx in enumerate(top_indices, start=1):
            rows.append(
                {
                    "topic_id": topic_idx,
                    "rank": rank,
                    "term": terms[term_idx],
                    "weight": float(component[term_idx]),
                }
            )

    return pd.DataFrame(rows)


def assign_topics(document_topic_matrix: np.ndarray) -> pd.DataFrame:
    """Assign each document to its highest-weighted topic."""
    if document_topic_matrix.ndim != 2:
        msg = "document_topic_matrix must be a 2D array."
        raise ValueError(msg)

    dominant_topic = document_topic_matrix.argmax(axis=1)
    dominant_score = document_topic_matrix.max(axis=1)

    return pd.DataFrame(
        {
            "topic_id": dominant_topic.astype(int),
            "topic_score": dominant_score.astype(float),
        }
    )
