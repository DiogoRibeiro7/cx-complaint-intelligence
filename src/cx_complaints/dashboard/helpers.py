"""Pure-Python helpers for the Streamlit dashboard.

These functions have no Streamlit dependency, which makes them importable and
unit-testable without ``streamlit`` installed.  The Streamlit-cached loaders
live in :mod:`cx_complaints.dashboard.loaders`; the pages import from both.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cx_complaints.reports.notebook_utils import (
    NotebookContext,
    find_project_root,
)

WEAK_RESPONSE_VALUES: tuple[str, ...] = (
    "closed",
    "closed with explanation",
    "in progress",
)


@dataclass(frozen=True)
class DashboardContext:
    """Filesystem locations needed by every dashboard page."""

    project_root: Path
    data_raw: Path
    pipeline_outputs: Path


def dashboard_context() -> DashboardContext:
    """Resolve project paths without writing any artefact directories.

    Unlike :func:`cx_complaints.reports.notebook_utils.setup_notebook`, this
    helper does not create a ``reports/notebook/<id>/`` folder &mdash; the
    dashboard is read-mostly and should not pollute the artefact tree just by
    being launched.
    """
    project_root = find_project_root()
    return DashboardContext(
        project_root=project_root,
        data_raw=project_root / "data" / "raw",
        pipeline_outputs=project_root / "reports" / "sample",
    )


def as_notebook_context(ctx: DashboardContext) -> NotebookContext:
    """Adapt a dashboard context to the shape the notebook loaders expect."""
    return NotebookContext(
        notebook_id="dashboard",
        project_root=ctx.project_root,
        data_raw=ctx.data_raw,
        reports_root=ctx.project_root / "reports",
        pipeline_outputs=ctx.pipeline_outputs,
        artefact_dir=ctx.project_root / "reports" / "notebook" / "dashboard",
    )


def resolve_topic_label(
    topic_id: int,
    labels: pd.DataFrame,
    *,
    fallback: str = "Unlabeled topic",
) -> str:
    """Resolve a topic label from ``labels`` with a safe fallback.

    This protects page-level formatters from crashing when pipeline outputs are
    partial or when a topic assignment exists without a corresponding label row.
    """
    if labels.empty or "topic_id" not in labels.columns or "topic_label" not in labels.columns:
        return fallback

    matches = labels.loc[labels["topic_id"] == topic_id, "topic_label"]
    if matches.empty:
        return fallback

    value = str(matches.iloc[0]).strip()
    return value if value else fallback


def apply_filters(
    frame: pd.DataFrame,
    *,
    products: list[str] | None = None,
    companies: list[str] | None = None,
    risk_levels: list[str] | None = None,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """Return ``frame`` filtered by the optional sidebar selections.

    Empty / ``None`` selections behave as "no filter" so callers do not need to
    special-case the unfiltered path.
    """
    filtered = frame
    if products:
        filtered = filtered[filtered["product"].fillna("Unknown").isin(products)]
    if companies:
        filtered = filtered[filtered["company"].fillna("Unknown").isin(companies)]
    if risk_levels:
        filtered = filtered[filtered["risk_level"].isin(risk_levels)]
    if date_range is not None:
        start, end = date_range
        dates = pd.to_datetime(filtered["date_received"], errors="coerce")
        filtered = filtered[(dates >= start) & (dates <= end)]
    return filtered


def priority_queue(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the high-risk + untimely + weak-response priority queue.

    The output is sorted by ``risk_score`` descending so the most urgent items
    appear first.
    """
    if frame.empty:
        return frame.copy()

    timely_is_yes = frame["timely"].fillna("").astype(str).str.strip().str.lower() == "yes"
    weak_response = (
        frame["company_response"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(WEAK_RESPONSE_VALUES)
    )
    is_high_risk = frame["risk_level"].fillna("").astype(str) == "high"
    queue = frame[is_high_risk & ~timely_is_yes & weak_response].copy()
    return queue.sort_values("risk_score", ascending=False)


def weekly_kpis(scores: pd.DataFrame) -> pd.DataFrame:
    """Aggregate scored complaints into per-week KPI rows."""
    if scores.empty:
        return pd.DataFrame(
            columns=[
                "week",
                "complaints",
                "high_risk",
                "avg_risk_score",
                "untimely",
                "high_risk_share_pct",
                "untimely_share_pct",
            ]
        )

    working = scores.copy()
    working["date_received"] = pd.to_datetime(working["date_received"], errors="coerce")
    working = working.dropna(subset=["date_received"])
    working["week"] = working["date_received"].dt.to_period("W").astype(str)

    aggregated = (
        working.groupby("week", as_index=False)
        .agg(
            complaints=("complaint_id", "size"),
            high_risk=("risk_level", lambda series: (series == "high").sum()),
            avg_risk_score=("risk_score", "mean"),
            untimely=(
                "timely",
                lambda series: (series.fillna("").str.lower() != "yes").sum(),
            ),
        )
        .sort_values("week")
        .reset_index(drop=True)
    )
    aggregated["high_risk_share_pct"] = (
        aggregated["high_risk"] / aggregated["complaints"] * 100
    ).round(1)
    aggregated["untimely_share_pct"] = (
        aggregated["untimely"] / aggregated["complaints"] * 100
    ).round(1)
    aggregated["avg_risk_score"] = aggregated["avg_risk_score"].round(2)
    return aggregated


def week_over_week(weekly: pd.DataFrame) -> pd.DataFrame:
    """Return a current vs. previous week delta table.

    Empty when fewer than two weeks of data are present.
    """
    if len(weekly) < 2:
        return pd.DataFrame(columns=["metric", "current", "previous", "delta"])

    current = weekly.iloc[-1]
    previous = weekly.iloc[-2]
    metrics = ["complaints", "high_risk_share_pct", "untimely_share_pct", "avg_risk_score"]
    rows = [
        {
            "metric": metric,
            "current": float(current[metric]),
            "previous": float(previous[metric]),
            "delta": round(float(current[metric]) - float(previous[metric]), 2),
        }
        for metric in metrics
    ]
    return pd.DataFrame(rows)


def topic_weekly_movement(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-topic week-over-week movement between the last two weeks."""
    if scores.empty or labels.empty:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "topic_label",
                "latest_complaints",
                "prev_complaints",
                "delta_complaints",
            ]
        )

    dated = scores.dropna(subset=["topic_id"]).copy()
    if dated.empty:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "topic_label",
                "latest_complaints",
                "prev_complaints",
                "delta_complaints",
            ]
        )

    dated["topic_id"] = dated["topic_id"].astype(int)
    dated["date_received"] = pd.to_datetime(dated["date_received"], errors="coerce")
    dated = dated.dropna(subset=["date_received"])
    dated["week"] = dated["date_received"].dt.to_period("W").astype(str)

    weeks = sorted(dated["week"].unique())
    if len(weeks) < 2:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "topic_label",
                "latest_complaints",
                "prev_complaints",
                "delta_complaints",
            ]
        )

    latest_week, prev_week = weeks[-1], weeks[-2]
    latest = (
        dated[dated["week"] == latest_week]
        .groupby("topic_id", as_index=False)
        .size()
        .rename(columns={"size": "latest_complaints"})
    )
    prev = (
        dated[dated["week"] == prev_week]
        .groupby("topic_id", as_index=False)
        .size()
        .rename(columns={"size": "prev_complaints"})
    )
    movement = (
        labels.merge(latest, on="topic_id", how="left")
        .merge(prev, on="topic_id", how="left")
        .fillna({"latest_complaints": 0, "prev_complaints": 0})
    )
    movement[["latest_complaints", "prev_complaints"]] = movement[
        ["latest_complaints", "prev_complaints"]
    ].astype(int)
    movement["delta_complaints"] = (
        movement["latest_complaints"] - movement["prev_complaints"]
    )
    return movement.sort_values("delta_complaints", ascending=False).reset_index(drop=True)


def build_executive_brief(
    weekly: pd.DataFrame,
    topic_move: pd.DataFrame,
    priority_queue_count: int,
    weekly_total: int,
) -> str:
    """Build the same Markdown brief that notebook 10 produces.

    Numbers come exclusively from the supplied tables. ``ValueError`` is raised
    if there is not enough history to compute deltas, mirroring the notebook.
    """
    if len(weekly) < 2:
        msg = "Executive brief requires at least two weeks of data."
        raise ValueError(msg)

    current = weekly.iloc[-1]
    previous = weekly.iloc[-2]
    delta = {
        "complaints": int(current["complaints"] - previous["complaints"]),
        "high_risk_share_pct": round(
            float(current["high_risk_share_pct"] - previous["high_risk_share_pct"]),
            1,
        ),
        "untimely_share_pct": round(
            float(current["untimely_share_pct"] - previous["untimely_share_pct"]),
            1,
        ),
        "avg_risk_score": round(
            float(current["avg_risk_score"] - previous["avg_risk_score"]),
            2,
        ),
    }
    queue_share = round(
        priority_queue_count / max(weekly_total, 1) * 100,
        1,
    )

    rising_lines = [
        (
            f"- Topic {int(row['topic_id'])} ({row['topic_label']}): "
            f"{int(row['delta_complaints']):+,} complaints WoW "
            f"(now {int(row['latest_complaints']):,}, was {int(row['prev_complaints']):,})."
        )
        for _, row in topic_move.head(3).iterrows()
    ]
    if not rising_lines:
        rising_lines = ["- No topic movement available for the current period."]

    return "\n".join(
        [
            f"# Weekly Executive Brief ({current['week']})",
            "",
            "## What changed",
            "",
            f"- Complaints: {int(current['complaints']):,} ({delta['complaints']:+,} WoW).",
            f"- High-risk share: {current['high_risk_share_pct']:.1f}% "
            f"({delta['high_risk_share_pct']:+.1f} pp WoW).",
            f"- Untimely share: {current['untimely_share_pct']:.1f}% "
            f"({delta['untimely_share_pct']:+.1f} pp WoW).",
            f"- Average risk score: {current['avg_risk_score']:.2f} "
            f"({delta['avg_risk_score']:+.2f} WoW).",
            "",
            "## Emerging topics",
            "",
            *rising_lines,
            "",
            "## Operational queue",
            "",
            (
                f"- Priority queue: {priority_queue_count:,} complaints "
                f"({queue_share:.1f}% of weekly volume)."
            ),
            "",
            "## Recommended actions",
            "",
            "1. Review the top priority-queue items within 24 hours.",
            "2. Assign topic owners and validate root causes for the rising topics.",
            "3. Track whether high-risk and untimely shares normalise next week.",
        ]
    )
