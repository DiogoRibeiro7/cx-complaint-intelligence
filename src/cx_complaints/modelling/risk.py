"""Interpretable complaint severity and risk scoring."""

from __future__ import annotations

import pandas as pd

HIGH_RISK_PRODUCTS = {
    "credit reporting or other personal consumer reports",
    "credit reporting, credit repair services, or other personal consumer reports",
    "debt collection",
    "mortgage",
    "payday loan, title loan, personal loan, or advance loan",
}

HIGH_RISK_ISSUE_TERMS = (
    "identity theft",
    "incorrect information",
    "attempts to collect debt not owed",
    "threat",
    "foreclosure",
    "fraud",
    "scam",
    "lawsuit",
    "credit reporting",
)

WEAK_RESPONSE_TERMS = (
    "closed with explanation",
    "closed",
    "in progress",
)

VULNERABLE_TAG_TERMS = (
    "older american",
    "servicemember",
)


def score_complaint_risk(frame: pd.DataFrame) -> pd.DataFrame:
    """Add transparent risk-score columns to complaint records.

    The score is a prioritisation signal for analysis. It is not a legal finding and
    should not be used as the only basis for public company ranking.
    """
    output = frame.copy()

    required_defaults = {
        "timely": "",
        "company_response": "",
        "product": "",
        "issue": "",
        "sub_issue": "",
        "tags": "",
        "word_count": 0,
        "urgency_term_count": 0,
        "has_money_language": False,
    }
    for column, default in required_defaults.items():
        if column not in output.columns:
            output[column] = default

    score = pd.Series(0.0, index=output.index)

    timely = output["timely"].fillna("").astype(str).str.lower()
    response = output["company_response"].fillna("").astype(str).str.lower()
    product = output["product"].fillna("").astype(str).str.lower()
    issue_text = (
        output["issue"].fillna("").astype(str) + " " + output["sub_issue"].fillna("").astype(str)
    ).str.lower()
    tags = output["tags"].fillna("").astype(str).str.lower()

    score += (timely != "yes").astype(float) * 2.0
    score += product.isin(HIGH_RISK_PRODUCTS).astype(float) * 1.5
    score += issue_text.map(_contains_high_risk_issue).astype(float) * 1.5
    score += response.map(_looks_like_weak_response).astype(float) * 0.75
    score += tags.map(_contains_vulnerable_tag).astype(float) * 1.0
    score += (
        pd.to_numeric(output["urgency_term_count"], errors="coerce")
        .fillna(0)
        .clip(0, 3)
        * 0.75
    )
    score += output["has_money_language"].fillna(False).astype(bool).astype(float) * 0.5
    score += (
        (pd.to_numeric(output["word_count"], errors="coerce").fillna(0) >= 250)
        .astype(float)
        * 0.5
    )

    output["risk_score"] = score.round(2)
    output["risk_level"] = output["risk_score"].map(risk_level_from_score)
    return output


def risk_level_from_score(score: float) -> str:
    """Map a numeric risk score to a human-readable level."""
    if score >= 5.0:
        return "high"
    if score >= 2.5:
        return "medium"
    return "low"


def _contains_high_risk_issue(text: str) -> bool:
    """Return whether an issue text contains known high-risk issue terms."""
    return any(term in text for term in HIGH_RISK_ISSUE_TERMS)


def _looks_like_weak_response(text: str) -> bool:
    """Return whether a company response may require deeper review."""
    return any(term in text for term in WEAK_RESPONSE_TERMS)


def _contains_vulnerable_tag(text: str) -> bool:
    """Return whether CFPB tags indicate a potentially vulnerable consumer group."""
    return any(term in text for term in VULNERABLE_TAG_TERMS)
