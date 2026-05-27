"""Priority queue page: filterable high-risk + untimely + weak-response list."""

from __future__ import annotations

import streamlit as st

from cx_complaints.dashboard.data import apply_filters, load_complaints, priority_queue

st.set_page_config(page_title="Priority Queue — CX Complaint Intelligence", layout="wide")
st.title("Priority queue")
st.caption(
    "Complaints that are high-risk **and** untimely **and** have a weak response code. "
    "Work top-down: the first row is the most urgent."
)

try:
    complaints = load_complaints()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Filters")
    product_options = sorted(complaints["product"].fillna("Unknown").unique())
    company_options = sorted(complaints["company"].fillna("Unknown").unique())
    selected_products = st.multiselect("Products", options=product_options)
    selected_companies = st.multiselect("Companies", options=company_options)

filtered = apply_filters(
    complaints,
    products=selected_products or None,
    companies=selected_companies or None,
)
queue = priority_queue(filtered)

# KPI strip ------------------------------------------------------------------
share = (len(queue) / max(len(filtered), 1)) * 100
c1, c2, c3 = st.columns(3)
c1.metric("Complaints in scope", f"{len(filtered):,}")
c2.metric("Priority queue size", f"{len(queue):,}", f"{share:.1f}% of scope")
c3.metric(
    "Distinct companies in queue",
    f"{queue['company'].nunique() if not queue.empty else 0:,}",
)

if queue.empty:
    st.success("No complaints meet the priority criteria with the current filters.")
    st.stop()

st.divider()

# Company / issue rollups ----------------------------------------------------
st.subheader("Where the queue concentrates")
roll_company = (
    queue.groupby("company", dropna=False, as_index=False)
    .size()
    .rename(columns={"size": "complaints"})
    .sort_values("complaints", ascending=False)
)
roll_issue = (
    queue.groupby("issue", dropna=False, as_index=False)
    .size()
    .rename(columns={"size": "complaints"})
    .sort_values("complaints", ascending=False)
)

col_company, col_issue = st.columns(2)
with col_company:
    st.markdown("**By company**")
    st.dataframe(roll_company.head(10), hide_index=True, use_container_width=True)
with col_issue:
    st.markdown("**By issue**")
    st.dataframe(roll_issue.head(10), hide_index=True, use_container_width=True)

st.divider()

# Queue table ----------------------------------------------------------------
st.subheader("Queued complaints")
queue_columns = [
    "complaint_id",
    "date_received",
    "company",
    "product",
    "issue",
    "company_response",
    "timely",
    "risk_score",
]
st.dataframe(
    queue.loc[:, queue_columns].reset_index(drop=True),
    use_container_width=True,
    hide_index=True,
    height=500,
)
st.download_button(
    "Download priority queue (CSV)",
    data=queue.to_csv(index=False).encode("utf-8"),
    file_name="priority_queue.csv",
    mime="text/csv",
)
