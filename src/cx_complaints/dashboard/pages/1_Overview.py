"""Overview page: headline KPIs, monthly trend, and risk distribution."""

from __future__ import annotations

import altair as alt
import streamlit as st

from cx_complaints.dashboard.data import apply_filters, load_complaints
from cx_complaints.reports.notebook_utils import kpi_table, monthly_summary

st.set_page_config(page_title="Overview — CX Complaint Intelligence", layout="wide")
st.title("Overview")
st.caption("Headline KPIs, monthly trend, and the risk-score distribution.")

try:
    complaints = load_complaints()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

# Sidebar filters ------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    products = sorted(complaints["product"].fillna("Unknown").unique())
    risk_levels = ["low", "medium", "high"]
    selected_products = st.multiselect("Products", options=products, default=[])
    selected_risk = st.multiselect("Risk levels", options=risk_levels, default=[])

filtered = apply_filters(
    complaints,
    products=selected_products or None,
    risk_levels=selected_risk or None,
)

st.caption(f"Showing **{len(filtered):,}** of {len(complaints):,} complaints after filters.")

if filtered.empty:
    st.warning("No complaints match the current filters.")
    st.stop()

# KPI cards ------------------------------------------------------------------
kpis = kpi_table(filtered)
total_row = kpis[kpis["metric"] == "Total complaints"].iloc[0]
high_row = kpis[kpis["metric"] == "High-risk complaints"].iloc[0]
medhigh_row = kpis[kpis["metric"] == "Medium or high-risk complaints"].iloc[0]
untimely_row = kpis[kpis["metric"] == "Untimely responses"].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total complaints", f"{int(total_row['value']):,}")
c2.metric(
    "High-risk",
    f"{int(high_row['value']):,}",
    f"{high_row['share_of_total_pct']:.1f}%",
)
c3.metric(
    "Medium or higher",
    f"{int(medhigh_row['value']):,}",
    f"{medhigh_row['share_of_total_pct']:.1f}%",
)
c4.metric(
    "Untimely",
    f"{int(untimely_row['value']):,}",
    f"{untimely_row['share_of_total_pct']:.1f}%",
)

st.divider()

# Monthly trend --------------------------------------------------------------
st.subheader("Monthly trend")
monthly = monthly_summary(filtered)

if monthly.empty:
    st.info("Not enough dated complaints to plot a monthly trend.")
else:
    monthly_long = monthly.rename(
        columns={
            "complaints": "Complaints",
            "high_risk_share_pct": "High-risk share (%)",
        }
    )
    line_volume = (
        alt.Chart(monthly_long)
        .mark_line(point=True, color="#1F3A5F")
        .encode(
            x=alt.X("month:N", title="Month"),
            y=alt.Y("Complaints:Q", title="Complaint count"),
            tooltip=["month", "Complaints"],
        )
    )
    line_risk = (
        alt.Chart(monthly_long)
        .mark_line(point=True, color="#C0392B")
        .encode(
            x=alt.X("month:N", title="Month"),
            y=alt.Y("High-risk share (%):Q", title="High-risk share (%)"),
            tooltip=["month", "High-risk share (%)"],
        )
    )
    chart = alt.layer(line_volume, line_risk).resolve_scale(y="independent")
    st.altair_chart(chart, use_container_width=True)

st.divider()

# Risk distribution ----------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Risk-level distribution")
    risk_counts = (
        filtered["risk_level"]
        .fillna("unknown")
        .value_counts()
        .reindex(["low", "medium", "high", "unknown"], fill_value=0)
        .reset_index()
    )
    risk_counts.columns = ["risk_level", "complaints"]
    palette = {"low": "#3A7D44", "medium": "#E07A5F", "high": "#C0392B", "unknown": "#B0B7BF"}
    risk_chart = (
        alt.Chart(risk_counts)
        .mark_bar()
        .encode(
            x=alt.X("risk_level:N", sort=["low", "medium", "high", "unknown"]),
            y="complaints:Q",
            color=alt.Color(
                "risk_level:N",
                scale=alt.Scale(
                    domain=list(palette),
                    range=list(palette.values()),
                ),
                legend=None,
            ),
            tooltip=["risk_level", "complaints"],
        )
    )
    st.altair_chart(risk_chart, use_container_width=True)

with right:
    st.subheader("Risk-score distribution")
    score_chart = (
        alt.Chart(filtered[["risk_score"]].dropna())
        .mark_bar(color="#1F3A5F")
        .encode(
            x=alt.X("risk_score:Q", bin=alt.Bin(maxbins=20), title="Risk score"),
            y=alt.Y("count():Q", title="Complaint count"),
        )
    )
    st.altair_chart(score_chart, use_container_width=True)

st.divider()

# Recent rows preview --------------------------------------------------------
st.subheader("Latest complaints in the filtered set")
preview_columns = [
    "complaint_id",
    "date_received",
    "product",
    "issue",
    "company",
    "risk_score",
    "risk_level",
    "timely",
]
preview = (
    filtered.sort_values("date_received", ascending=False)
    .head(25)
    .loc[:, preview_columns]
    .reset_index(drop=True)
)
st.dataframe(preview, use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered complaints (CSV)",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_complaints.csv",
    mime="text/csv",
)
