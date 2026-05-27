"""Concentration page: top products, issues, and companies."""

from __future__ import annotations

from typing import cast

import altair as alt
import pandas as pd
import streamlit as st

from cx_complaints.dashboard.data import apply_filters, load_complaints
from cx_complaints.reports.notebook_utils import concentration_table

st.set_page_config(page_title="Concentration — CX Complaint Intelligence", layout="wide")
st.title("Concentration")
st.caption(
    "Where complaints concentrate by product, issue, and company. "
    "Adjust the minimum-volume filter to keep company comparisons stable."
)

try:
    complaints = load_complaints()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Filters")
    top_n = st.slider("Top-N rows", min_value=5, max_value=25, value=10, step=1)
    min_volume = st.slider(
        "Minimum company complaint count",
        min_value=0,
        max_value=100,
        value=20,
        step=5,
        help="Companies with fewer complaints are excluded from the risk-share ranking.",
    )
    risk_levels = st.multiselect(
        "Risk levels", options=["low", "medium", "high"], default=[]
    )

filtered = apply_filters(complaints, risk_levels=risk_levels or None)
st.caption(
    f"Showing **{len(filtered):,}** of {len(complaints):,} complaints after filters."
)

if filtered.empty:
    st.warning("No complaints match the current filters.")
    st.stop()

# Side-by-side bar charts ----------------------------------------------------
left, right = st.columns(2)


def _bar_chart(
    frame: pd.DataFrame,
    *,
    x: str,
    y: str,
    x_title: str,
    color: str,
) -> alt.Chart:
    chart = (
        alt.Chart(frame)
        .mark_bar(color=color)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title),
            y=alt.Y(f"{y}:N", sort="-x", title=None),
            tooltip=list(frame.columns),
        )
        .properties(height=350)
    )
    return cast(alt.Chart, chart)


with left:
    st.subheader("Top products by volume")
    products = concentration_table(filtered, "product", top_n=top_n)
    st.altair_chart(
        _bar_chart(
            products,
            x="complaints",
            y="product",
            x_title="Complaint count",
            color="#1F3A5F",
        ),
        use_container_width=True,
    )
    st.dataframe(products, hide_index=True, use_container_width=True)

with right:
    st.subheader("Top issues by volume")
    issues = concentration_table(filtered, "issue", top_n=top_n)
    st.altair_chart(
        _bar_chart(
            issues,
            x="complaints",
            y="issue",
            x_title="Complaint count",
            color="#4F86C6",
        ),
        use_container_width=True,
    )
    st.dataframe(issues, hide_index=True, use_container_width=True)

st.divider()

# Companies ------------------------------------------------------------------
st.subheader("Companies")
st.markdown(
    "Two views: highest **volume** (regardless of size) on the left, highest "
    "**high-risk share** (with the minimum-volume guardrail applied) on the right."
)

companies_volume = concentration_table(filtered, "company", top_n=top_n)
companies_risk = concentration_table(
    filtered, "company", top_n=top_n, min_volume=min_volume
)
companies_risk = companies_risk.sort_values(
    "high_risk_share_pct", ascending=False
).reset_index(drop=True)

col_volume, col_risk = st.columns(2)

with col_volume:
    st.markdown("**Highest volume**")
    st.altair_chart(
        _bar_chart(
            companies_volume,
            x="complaints",
            y="company",
            x_title="Complaint count",
            color="#1F3A5F",
        ),
        use_container_width=True,
    )
    st.dataframe(companies_volume, hide_index=True, use_container_width=True)

with col_risk:
    st.markdown(f"**Highest high-risk share (>= {min_volume} complaints)**")
    if companies_risk.empty:
        st.info("No companies cleared the minimum-volume filter. Lower the threshold.")
    else:
        st.altair_chart(
            _bar_chart(
                companies_risk,
                x="high_risk_share_pct",
                y="company",
                x_title="High-risk share (%)",
                color="#C0392B",
            ),
            use_container_width=True,
        )
        st.dataframe(companies_risk, hide_index=True, use_container_width=True)
