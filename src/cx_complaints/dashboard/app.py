"""Landing page for the CX Complaint Intelligence dashboard.

Streamlit auto-discovers other pages from the ``pages/`` directory next to
this file.  Run with::

    poetry run streamlit run src/cx_complaints/dashboard/app.py
"""

from __future__ import annotations

import streamlit as st

from cx_complaints.dashboard.helpers import dashboard_context
from cx_complaints.dashboard.loaders import load_complaints, load_pipeline_outputs

st.set_page_config(
    page_title="CX Complaint Intelligence",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("CX Complaint Intelligence")
st.caption(
    "Interactive companion to the analysis notebooks. "
    "All numbers come from the same pipeline outputs."
)

ctx = dashboard_context()

# Headline status panel ------------------------------------------------------
status_col, paths_col = st.columns([1, 2])

with status_col:
    st.subheader("Data sources")
    try:
        complaints = load_complaints()
        st.success(f"Complaints sample loaded: {len(complaints):,} rows.")
    except FileNotFoundError as exc:
        complaints = None
        st.error(str(exc))

    try:
        scores, _, labels = load_pipeline_outputs()
        st.success(
            f"Pipeline outputs loaded: {len(scores):,} scored complaints "
            f"across {len(labels)} topics."
        )
    except FileNotFoundError as exc:
        scores = None
        labels = None
        st.warning(str(exc))

with paths_col:
    st.subheader("Project paths")
    st.code(
        f"project root        {ctx.project_root}\n"
        f"data/raw            {ctx.data_raw}\n"
        f"pipeline outputs    {ctx.pipeline_outputs}",
        language="text",
    )

st.divider()

# Navigation cheatsheet ------------------------------------------------------
st.subheader("Pages")
st.markdown(
    """
    Use the sidebar to switch between views:

    | Page | What it answers |
    |------|-----------------|
    | **Overview** | Headline KPIs, monthly trend, risk distribution. |
    | **Concentration** | Top products, issues, and companies by volume and risk share. |
    | **Topics** | Topic-level risk concentration with representative complaints. |
    | **Priority Queue** | Filterable list of high-risk + untimely + weak-response complaints. |
    | **Weekly Brief** | Weekly KPIs, WoW deltas, and the auto-generated executive brief. |
    """
)

st.divider()

# How to refresh -------------------------------------------------------------
st.subheader("Refreshing the data")
st.markdown(
    """
    The dashboard reads two folders under the project root:

    - `data/raw/cfpb_sample.csv` &mdash; the raw complaints sample.
    - `reports/sample/complaint_scores.csv` and `topic_terms.csv` &mdash; the
      outputs of `cx-complaints analyze`.

    To refresh after a new fetch:

    ```bash
    poetry run cx-complaints fetch --output data/raw/cfpb_sample.csv
    poetry run cx-complaints analyze \\
        --input data/raw/cfpb_sample.csv \\
        --output-dir reports/sample
    ```

    Then click **C** in the top right of the dashboard (or restart Streamlit)
    to clear the in-memory cache.
    """
)
