from __future__ import annotations

from pathlib import Path

import pandas as pd

from cx_complaints.reports.summary import build_executive_summary, write_report_outputs
from cx_complaints.reports.tables import topic_distribution, value_summary


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product": ["Credit card", "Mortgage", "Credit card"],
            "issue": ["Billing", "Foreclosure", "Billing"],
            "company": ["A", "B", "A"],
            "topic_id": [0, 1, 0],
            "risk_score": [6.0, 2.0, 4.0],
            "risk_level": ["high", "low", "medium"],
        }
    )


def _sample_topic_terms() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "topic_id": [0, 0, 1, 1],
            "rank": [1, 2, 1, 2],
            "term": ["billing", "charge", "mortgage", "foreclosure"],
            "weight": [0.5, 0.4, 0.6, 0.3],
        }
    )


def test_value_summary_computes_counts_and_risk_metrics() -> None:
    result = value_summary(_sample_frame(), "product")

    assert list(result.columns) == [
        "product",
        "complaints",
        "average_risk_score",
        "high_risk_share",
    ]
    credit_card = result.loc[result["product"] == "Credit card"].iloc[0]
    assert int(credit_card["complaints"]) == 2
    assert float(credit_card["average_risk_score"]) == 5.0
    assert float(credit_card["high_risk_share"]) == 50.0


def test_topic_distribution_computes_topic_summary() -> None:
    result = topic_distribution(_sample_frame())

    topic_zero = result.loc[result["topic_id"] == 0].iloc[0]
    assert int(topic_zero["complaints"]) == 2
    assert float(topic_zero["average_risk_score"]) == 5.0
    assert float(topic_zero["high_risk_share"]) == 50.0


def test_build_executive_summary_includes_core_sections() -> None:
    frame = _sample_frame()
    topic_terms = _sample_topic_terms()
    summary = build_executive_summary(
        frame=frame,
        topic_terms=topic_terms,
        product_summary=value_summary(frame, "product"),
        issue_summary=value_summary(frame, "issue"),
        company_summary=value_summary(frame, "company"),
        topic_summary=topic_distribution(frame),
    )

    assert "# CFPB Complaint Intelligence Executive Summary" in summary
    assert "- Total complaints analysed: **3**." in summary
    assert "- High-risk complaints: **1** (33.3%)." in summary
    assert "## Recommended next actions" in summary
    assert "- **Topic 0**: billing, charge" in summary


def test_write_report_outputs_writes_expected_files(tmp_path: Path) -> None:
    output_dir = tmp_path / "reports_out"
    write_report_outputs(
        frame=_sample_frame(),
        topic_terms=_sample_topic_terms(),
        output_dir=output_dir,
    )

    expected_files = {
        "complaint_scores.csv",
        "topic_terms.csv",
        "product_summary.csv",
        "issue_summary.csv",
        "company_summary.csv",
        "topic_distribution.csv",
        "executive_summary.md",
    }
    written = {path.name for path in output_dir.iterdir()}
    assert expected_files == written
