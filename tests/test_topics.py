from __future__ import annotations

import pandas as pd

from cx_complaints.nlp.topics import fit_nmf_topics


def test_fit_nmf_topics_returns_assignments() -> None:
    texts = pd.Series(
        [
            "credit report identity theft wrong account",
            "credit report wrong balance identity theft",
            "mortgage payment escrow servicing problem",
            "mortgage escrow payment servicing issue",
            "debt collection harassment phone calls",
            "debt collector repeated harassment calls",
        ]
    )

    result = fit_nmf_topics(texts, n_topics=2, min_df=1, max_features=100)

    assert len(result.assignments) == len(texts)
    assert set(result.topic_terms.columns) == {"topic_id", "rank", "term", "weight"}
