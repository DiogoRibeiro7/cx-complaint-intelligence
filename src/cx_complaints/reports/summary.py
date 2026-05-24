"""Markdown executive summary generation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from cx_complaints.reports.tables import topic_distribution, value_summary


def write_report_outputs(
    frame: pd.DataFrame,
    topic_terms: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    """Write all report artifacts to disk."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    frame.to_csv(directory / "complaint_scores.csv", index=False)
    topic_terms.to_csv(directory / "topic_terms.csv", index=False)

    product_summary = value_summary(frame, "product")
    issue_summary = value_summary(frame, "issue")
    company_summary = value_summary(frame, "company")
    topic_summary = topic_distribution(frame)

    product_summary.to_csv(directory / "product_summary.csv", index=False)
    issue_summary.to_csv(directory / "issue_summary.csv", index=False)
    company_summary.to_csv(directory / "company_summary.csv", index=False)
    topic_summary.to_csv(directory / "topic_distribution.csv", index=False)

    markdown = build_executive_summary(
        frame=frame,
        topic_terms=topic_terms,
        product_summary=product_summary,
        issue_summary=issue_summary,
        company_summary=company_summary,
        topic_summary=topic_summary,
    )
    (directory / "executive_summary.md").write_text(markdown, encoding="utf-8")


def build_executive_summary(
    frame: pd.DataFrame,
    topic_terms: pd.DataFrame,
    product_summary: pd.DataFrame,
    issue_summary: pd.DataFrame,
    company_summary: pd.DataFrame,
    topic_summary: pd.DataFrame,
) -> str:
    """Build a business-readable Markdown summary."""
    total = len(frame)
    high_risk = int((frame["risk_level"] == "high").sum()) if total else 0
    high_risk_share = (high_risk / total * 100) if total else 0.0
    medium_or_high = int(frame["risk_level"].isin(["medium", "high"]).sum()) if total else 0

    lines: list[str] = [
        "# CFPB Complaint Intelligence Executive Summary",
        "",
        "## Overview",
        "",
        f"- Total complaints analysed: **{total:,}**.",
        f"- High-risk complaints: **{high_risk:,}** ({high_risk_share:.1f}%).",
        f"- Medium or high-risk complaints: **{medium_or_high:,}**.",
        "",
        "The risk score is an interpretable prioritisation signal. It combines response timeliness, "
        "product and issue risk, urgency language, monetary language, narrative length, and CFPB tags.",
        "",
    ]

    lines.extend(_table_section("Top products", product_summary))
    lines.extend(_table_section("Top issues", issue_summary))
    lines.extend(_table_section("Top companies", company_summary))
    lines.extend(_table_section("Topic distribution", topic_summary))
    lines.extend(_topic_terms_section(topic_terms))

    lines.extend(
        [
            "## Recommended next actions",
            "",
            "1. Review high-risk complaints first and validate whether the score aligns with analyst judgement.",
            "2. Use topic groups to identify recurring process failures, not only isolated customer issues.",
            "3. Compare companies only after normalising by exposure, customer base, or complaint volume context.",
            "4. Track topic and issue trends over time to detect emerging customer harm earlier.",
            "5. Convert validated high-risk labels into a supervised model once enough reviewed examples exist.",
            "",
            "## Limitations",
            "",
            "- The score is heuristic and should not be interpreted as a legal or regulatory judgement.",
            "- CFPB narratives are available only when consumers consent to publication.",
            "- Company comparisons require external normalisation before strong conclusions are made.",
        ]
    )

    return "\n".join(lines).strip() + "\n"


def _table_section(title: str, table: pd.DataFrame, max_rows: int = 10) -> list[str]:
    """Render a compact Markdown table section."""
    if table.empty:
        return [f"## {title}", "", "No rows available.", ""]

    return [f"## {title}", "", table.head(max_rows).to_markdown(index=False), ""]


def _topic_terms_section(topic_terms: pd.DataFrame) -> list[str]:
    """Render topic terms as bullet points."""
    lines = ["## Topic labels from top terms", ""]
    if topic_terms.empty:
        lines.extend(["No topic terms available.", ""])
        return lines

    for topic_id, group in topic_terms.groupby("topic_id"):
        terms = ", ".join(group.sort_values("rank")["term"].head(8).astype(str).tolist())
        lines.append(f"- **Topic {topic_id}**: {terms}")
    lines.append("")
    return lines
