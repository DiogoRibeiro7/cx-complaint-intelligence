"""Weekly brief page: weekly KPIs, WoW deltas, and the auto-generated brief."""

from __future__ import annotations

import altair as alt
import streamlit as st

from cx_complaints.dashboard.helpers import (
    build_executive_brief,
    priority_queue,
    topic_weekly_movement,
    week_over_week,
    weekly_kpis,
)
from cx_complaints.dashboard.loaders import load_pipeline_outputs

st.set_page_config(page_title="Weekly Brief — CX Complaint Intelligence", layout="wide")
st.title("Weekly brief")
st.caption(
    "Live version of notebook 09 + notebook 10. Numbers come from the computed "
    "weekly KPI table &mdash; nothing is hand-written."
)

try:
    scores, _, labels = load_pipeline_outputs()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Run `poetry run cx-complaints analyze --input data/raw/cfpb_sample.csv "
        "--output-dir reports/sample` first."
    )
    st.stop()

weekly = weekly_kpis(scores)
if weekly.empty:
    st.warning("No weekly data could be computed from the pipeline outputs.")
    st.stop()

# Latest week KPIs -----------------------------------------------------------
latest = weekly.iloc[-1]
st.subheader(f"Week of {latest['week']}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Complaints", f"{int(latest['complaints']):,}")
c2.metric("High-risk share", f"{latest['high_risk_share_pct']:.1f}%")
c3.metric("Untimely share", f"{latest['untimely_share_pct']:.1f}%")
c4.metric("Avg risk score", f"{latest['avg_risk_score']:.2f}")

# WoW deltas -----------------------------------------------------------------
deltas = week_over_week(weekly)
if deltas.empty:
    st.info("Only one week of data &mdash; week-over-week deltas not available.")
else:
    st.markdown("**Week-over-week change**")
    st.dataframe(deltas, hide_index=True, use_container_width=True)

st.divider()

# Trend charts ---------------------------------------------------------------
st.subheader("Weekly trends")
trend_volume = (
    alt.Chart(weekly)
    .transform_fold(
        ["complaints", "high_risk"], as_=["series", "value"]
    )
    .mark_line(point=True)
    .encode(
        x=alt.X("week:N", title="Week"),
        y=alt.Y("value:Q", title="Count"),
        color=alt.Color("series:N", legend=alt.Legend(title=None)),
        tooltip=["week", "series:N", "value:Q"],
    )
    .properties(height=320)
)
trend_share = (
    alt.Chart(weekly)
    .transform_fold(
        ["high_risk_share_pct", "untimely_share_pct"], as_=["series", "value"]
    )
    .mark_line(point=True)
    .encode(
        x=alt.X("week:N", title="Week"),
        y=alt.Y("value:Q", title="Share (%)"),
        color=alt.Color("series:N", legend=alt.Legend(title=None)),
        tooltip=["week", "series:N", "value:Q"],
    )
    .properties(height=320)
)
chart_left, chart_right = st.columns(2)
chart_left.altair_chart(trend_volume, use_container_width=True)
chart_right.altair_chart(trend_share, use_container_width=True)

st.divider()

# Topic movement -------------------------------------------------------------
topic_move = topic_weekly_movement(scores, labels)
st.subheader("Topic movement this week")
if topic_move.empty:
    st.info("Need at least two weeks of data to compute topic movement.")
else:
    st.dataframe(topic_move.head(10), hide_index=True, use_container_width=True)

st.divider()

# Executive brief ------------------------------------------------------------
st.subheader("Generated executive brief")
queue = priority_queue(scores)
queue_week = (
    queue[queue["date_received"].dt.to_period("W").astype(str) == latest["week"]]
    if "date_received" in queue.columns and not queue.empty
    else queue
)

try:
    brief = build_executive_brief(
        weekly=weekly,
        topic_move=topic_move,
        priority_queue_count=int(len(queue_week)),
        weekly_total=int(latest["complaints"]),
    )
    st.markdown(brief)
    st.download_button(
        "Download brief (Markdown)",
        data=brief.encode("utf-8"),
        file_name=f"weekly_executive_brief_{latest['week']}.md",
        mime="text/markdown",
    )
except ValueError as exc:
    st.warning(str(exc))
