"""Smoke tests for the Streamlit dashboard pages.

Uses ``streamlit.testing.v1.AppTest`` to render each page in-process. The
tests do not assert on visual layout &mdash; they assert that:

1. each page imports and runs without raising;
2. the expected title is set;
3. no exception is captured by Streamlit's runtime.

The dashboard reads the bundled sample under ``data/raw/cfpb_sample.csv`` and
the pipeline outputs under ``reports/sample/``. These tests skip themselves
gracefully if either is missing in the test environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

streamlit_testing = pytest.importorskip("streamlit.testing.v1")

if TYPE_CHECKING:
    from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "src" / "cx_complaints" / "dashboard"
SAMPLE_DATA = PROJECT_ROOT / "data" / "raw" / "cfpb_sample.csv"
PIPELINE_OUTPUTS = PROJECT_ROOT / "reports" / "sample" / "complaint_scores.csv"


pytestmark = pytest.mark.skipif(
    not SAMPLE_DATA.exists(),
    reason="Sample data not available; run cx-complaints fetch first.",
)


def _run(path: Path, *, timeout: float = 30.0) -> AppTest:
    """Run a Streamlit script and return the AppTest result.

    Raises if the script raises a Python exception. Streamlit's own captured
    errors are checked separately so the assertion message is clearer.
    """
    app: AppTest = streamlit_testing.AppTest.from_file(
        str(path),
        default_timeout=timeout,
    )
    app.run()
    if app.exception:
        raise AssertionError(
            f"Streamlit captured an exception while running {path.name}: "
            f"{app.exception[0].value}"
        )
    return app


def test_home_page_renders() -> None:
    """app.py should render the landing-page title."""
    app = _run(DASHBOARD_DIR / "app.py")
    titles = [block.value for block in app.title]
    assert any("CX Complaint Intelligence" in title for title in titles)


def test_overview_page_renders() -> None:
    """Overview page renders the KPI columns even with default filters."""
    app = _run(DASHBOARD_DIR / "pages" / "1_Overview.py")
    titles = [block.value for block in app.title]
    assert any("Overview" in title for title in titles)
    assert len(app.metric) >= 4, "Expected four KPI cards on the Overview page."


def test_concentration_page_renders() -> None:
    """Concentration page renders the volume/risk side-by-side leaderboards."""
    app = _run(DASHBOARD_DIR / "pages" / "2_Concentration.py")
    titles = [block.value for block in app.title]
    assert any("Concentration" in title for title in titles)


@pytest.mark.skipif(
    not PIPELINE_OUTPUTS.exists(),
    reason="Pipeline outputs not available; run cx-complaints analyze first.",
)
def test_topics_page_renders() -> None:
    """Topics page renders the leaderboard and the bubble chart."""
    app = _run(DASHBOARD_DIR / "pages" / "3_Topics.py", timeout=60.0)
    titles = [block.value for block in app.title]
    assert any("Topics" in title for title in titles)


def test_priority_queue_page_renders() -> None:
    """Priority queue page renders the KPI strip and the queue table."""
    app = _run(DASHBOARD_DIR / "pages" / "4_Priority_Queue.py")
    titles = [block.value for block in app.title]
    assert any("Priority queue" in title for title in titles)
    assert len(app.metric) >= 3, "Expected three KPI cards on the Priority queue page."


@pytest.mark.skipif(
    not PIPELINE_OUTPUTS.exists(),
    reason="Pipeline outputs not available; run cx-complaints analyze first.",
)
def test_weekly_brief_page_renders() -> None:
    """Weekly brief page renders the brief markdown when ≥2 weeks of data exist."""
    app = _run(DASHBOARD_DIR / "pages" / "5_Weekly_Brief.py", timeout=60.0)
    titles = [block.value for block in app.title]
    assert any("Weekly brief" in title for title in titles)
