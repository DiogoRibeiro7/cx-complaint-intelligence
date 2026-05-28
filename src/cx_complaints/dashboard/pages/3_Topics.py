"""Topics page: topic-level risk concentration and trends."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from cx_complaints.dashboard.helpers import resolve_topic_label
from cx_complaints.dashboard.loaders import load_pipeline_outputs

st.set_page_config(page_title="Topics — CX Complaint Intelligence", layout="wide")
st.title("Topics")
st.caption(
    "NMF topic assignments from the analysis pipeline. "
    "Bubble size scales with topic volume; colour scales with high-risk share."
)

try:
    scores, _, labels = load_pipeline_outputs()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.info(
        "Run `poetry run cx-complaints analyze --input data/raw/cfpb_sample.csv "
        "--output-dir reports/sample` to produce the pipeline outputs this page needs."
    )
    st.stop()

scores = scores.dropna(subset=["topic_id"]).copy()
scores["topic_id"] = scores["topic_id"].astype(int)

# Topic summary table --------------------------------------------------------
summary = (
    scores.groupby("topic_id", as_index=False)
    .agg(
        complaints=("complaint_id", "size"),
        avg_risk=("risk_score", "mean"),
        high_risk_share_pct=("risk_level", lambda s: (s == "high").mean() * 100),
    )
    .merge(labels, on="topic_id", how="left")
    .sort_values(["high_risk_share_pct", "complaints"], ascending=[False, False])
    .reset_index(drop=True)
)
summary["avg_risk"] = summary["avg_risk"].round(2)
summary["high_risk_share_pct"] = summary["high_risk_share_pct"].round(1)

st.subheader("Topic leaderboard")
st.dataframe(summary, use_container_width=True, hide_index=True)

st.divider()

# Bubble chart ---------------------------------------------------------------
st.subheader("Volume vs. risk concentration")
bubble = (
    alt.Chart(summary)
    .mark_circle(opacity=0.85)
    .encode(
        x=alt.X("complaints:Q", title="Complaint count"),
        y=alt.Y("high_risk_share_pct:Q", title="High-risk share (%)"),
        size=alt.Size("complaints:Q", legend=None, scale=alt.Scale(range=[80, 1200])),
        color=alt.Color(
            "high_risk_share_pct:Q",
            scale=alt.Scale(scheme="orangered"),
            legend=alt.Legend(title="High-risk %"),
        ),
        tooltip=[
            "topic_id",
            "topic_label",
            "complaints",
            "avg_risk",
            "high_risk_share_pct",
        ],
    )
    .properties(height=420)
)
labels_layer = (
    alt.Chart(summary)
    .mark_text(dy=-12, fontSize=11)
    .encode(
        x="complaints:Q",
        y="high_risk_share_pct:Q",
        text=alt.Text("topic_id:N"),
    )
)
st.altair_chart(bubble + labels_layer, use_container_width=True)

st.divider()

# Topic deep-dive ------------------------------------------------------------
st.subheader("Topic deep dive")

topic_options = summary["topic_id"].tolist()
selected_topic = st.selectbox(
    "Pick a topic",
    options=topic_options,
    format_func=lambda topic_id: (
        f"Topic {topic_id} — "
        f"{resolve_topic_label(topic_id, labels)}"
    ),
)

topic_rows = scores[scores["topic_id"] == selected_topic].copy()
topic_rows["narrative_preview"] = (
    topic_rows["complaint_what_happened"].fillna("").astype(str).str.slice(0, 240) + "…"
)
preview_columns = [
    "complaint_id",
    "date_received",
    "product",
    "issue",
    "company",
    "risk_score",
    "risk_level",
    "narrative_preview",
]
top_samples = (
    topic_rows.sort_values("risk_score", ascending=False).head(5).loc[:, preview_columns]
)

st.markdown(
    f"**Topic {selected_topic}** contains {len(topic_rows):,} complaints. "
    f"Showing the five highest-risk examples."
)
st.dataframe(top_samples, hide_index=True, use_container_width=True)

# Topic trend over time ------------------------------------------------------
st.subheader("Topic share over time")
trend_source = scores.copy()
trend_source["date_received"] = pd.to_datetime(
    trend_source["date_received"], errors="coerce"
)
trend_source = trend_source.dropna(subset=["date_received"])
trend_source["month"] = trend_source["date_received"].dt.to_period("M").astype(str)

monthly_counts = (
    trend_source.groupby(["month", "topic_id"], as_index=False)
    .size()
    .rename(columns={"size": "complaints"})
)
month_totals = monthly_counts.groupby("month")["complaints"].sum().rename("month_total")
monthly = monthly_counts.merge(month_totals, on="month", how="left")
monthly["share_pct"] = (monthly["complaints"] / monthly["month_total"] * 100).round(2)
monthly = monthly.merge(labels, on="topic_id", how="left")

if monthly.empty:
    st.info("No monthly data available for the trend chart.")
else:
    trend_chart = (
        alt.Chart(monthly)
        .mark_line(point=True)
        .encode(
            x=alt.X("month:N", title="Month"),
            y=alt.Y("share_pct:Q", title="Topic share (%)"),
            color=alt.Color("topic_id:N", legend=alt.Legend(title="Topic")),
            tooltip=["month", "topic_id", "topic_label", "share_pct", "complaints"],
        )
        .properties(height=380)
    )
    st.altair_chart(trend_chart, use_container_width=True)
