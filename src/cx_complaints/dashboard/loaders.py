"""Streamlit-cached loaders for the dashboard.

These thin wrappers add ``@st.cache_data`` on top of the notebook-utility
loaders so the same parsed frame is reused across page interactions. They live
in their own module so the pure helpers in :mod:`cx_complaints.dashboard.helpers`
can be imported without ``streamlit`` installed.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from cx_complaints.dashboard.helpers import as_notebook_context, dashboard_context
from cx_complaints.reports.notebook_utils import (
    load_complaints as _load_complaints,
)
from cx_complaints.reports.notebook_utils import (
    load_pipeline_outputs as _load_pipeline_outputs,
)
from cx_complaints.reports.notebook_utils import (
    topic_labels,
)


@st.cache_data(show_spinner="Loading complaints sample…")
def load_complaints() -> pd.DataFrame:
    """Load, normalise, clean, and score the bundled complaints sample."""
    return _load_complaints(as_notebook_context(dashboard_context()))


@st.cache_data(show_spinner="Loading pipeline outputs…")
def load_pipeline_outputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the CLI ``analyze`` outputs plus a topic-label lookup."""
    scores, terms = _load_pipeline_outputs(as_notebook_context(dashboard_context()))
    labels = topic_labels(terms, top_n=4)
    return scores, terms, labels
