"""Summary tables for complaint intelligence reports."""

from __future__ import annotations

import pandas as pd


def value_summary(frame: pd.DataFrame, column: str, top_n: int = 15) -> pd.DataFrame:
    """Summarise counts and average risk by a categorical column."""
    if column not in frame.columns:
        msg = f"Missing required column: {column!r}."
        raise KeyError(msg)

    grouped = (
        frame.groupby(column, dropna=False)
        .agg(
            complaints=(column, "size"),
            average_risk_score=("risk_score", "mean"),
            high_risk_share=("risk_level", lambda values: (values == "high").mean()),
        )
        .reset_index()
        .sort_values(["complaints", "average_risk_score"], ascending=[False, False])
        .head(top_n)
    )
    grouped["average_risk_score"] = grouped["average_risk_score"].round(2)
    grouped["high_risk_share"] = (grouped["high_risk_share"] * 100).round(1)
    return grouped


def topic_distribution(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise complaint counts and risk by discovered topic."""
    if "topic_id" not in frame.columns:
        msg = "Missing topic assignments. Expected a 'topic_id' column."
        raise KeyError(msg)

    grouped = (
        frame.groupby("topic_id", dropna=False)
        .agg(
            complaints=("topic_id", "size"),
            average_risk_score=("risk_score", "mean"),
            high_risk_share=("risk_level", lambda values: (values == "high").mean()),
        )
        .reset_index()
        .sort_values("complaints", ascending=False)
    )
    grouped["average_risk_score"] = grouped["average_risk_score"].round(2)
    grouped["high_risk_share"] = (grouped["high_risk_share"] * 100).round(1)
    return grouped
