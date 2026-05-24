"""Feature engineering for complaint narratives and metadata."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

URGENCY_TERMS = (
    "fraud",
    "scam",
    "identity theft",
    "lawsuit",
    "sue",
    "court",
    "eviction",
    "foreclosure",
    "harassment",
    "threat",
    "urgent",
    "emergency",
    "ruined",
    "illegal",
)

MONEY_PATTERN = re.compile(r"\[MONEY\]|\$|dollar|dollars|fee|fees|charge|charged|refund", re.IGNORECASE)


def add_text_features(frame: pd.DataFrame, text_column: str = "clean_text") -> pd.DataFrame:
    """Add transparent text features used by reporting and risk scoring."""
    if text_column not in frame.columns:
        msg = f"Missing required text column: {text_column!r}."
        raise KeyError(msg)

    output = frame.copy()
    text = output[text_column].fillna("").astype(str)
    lower_text = text.str.lower()

    output["char_count"] = text.str.len().astype(int)
    output["word_count"] = text.str.split().map(len).astype(int)
    output["has_money_language"] = text.map(lambda value: bool(MONEY_PATTERN.search(value)))
    output["urgency_term_count"] = lower_text.map(_count_urgency_terms).astype(int)
    output["narrative_available"] = output["word_count"] > 0
    return output


def _count_urgency_terms(text: str) -> int:
    """Count urgency terms in a lower-cased narrative."""
    return int(sum(term in text for term in URGENCY_TERMS))


def safe_log1p(series: pd.Series) -> pd.Series:
    """Return log(1+x) while handling missing and negative values defensively."""
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    clipped = np.clip(numeric, a_min=0.0, a_max=None)
    return pd.Series(np.log1p(clipped), index=series.index)
