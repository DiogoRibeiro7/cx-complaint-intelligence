"""Text cleaning utilities for complaint narratives."""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

WHITESPACE_RE = re.compile(r"\s+")
REDACTION_RE = re.compile(r"\bX{2,}\b", flags=re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
MONEY_RE = re.compile(r"\$\s?\d+(?:[,.]\d+)*")


def normalize_text(value: object) -> str:
    """Clean a complaint narrative while preserving its semantic content.

    CFPB narratives often contain redaction placeholders such as ``XXXX``. We keep a
    generic ``[REDACTED]`` marker rather than removing it entirely, because repeated
    redactions can be informative about the structure of the complaint.
    """
    if value is None or pd.isna(value):
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" [URL] ", text)
    text = EMAIL_RE.sub(" [EMAIL] ", text)
    text = MONEY_RE.sub(" [MONEY] ", text)
    text = REDACTION_RE.sub(" [REDACTED] ", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def add_clean_text_column(
    frame: pd.DataFrame,
    source_column: str = "complaint_what_happened",
    target_column: str = "clean_text",
) -> pd.DataFrame:
    """Return a DataFrame with a cleaned narrative text column."""
    if source_column not in frame.columns:
        msg = f"Missing required source column: {source_column!r}."
        raise KeyError(msg)

    output = frame.copy()
    output[target_column] = output[source_column].map(normalize_text)
    return output
