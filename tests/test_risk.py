from __future__ import annotations

import pandas as pd

from cx_complaints.modelling.risk import risk_level_from_score, score_complaint_risk


def test_risk_level_from_score() -> None:
    assert risk_level_from_score(0.0) == "low"
    assert risk_level_from_score(2.5) == "medium"
    assert risk_level_from_score(5.0) == "high"


def test_score_complaint_risk_flags_high_risk_complaint() -> None:
    frame = pd.DataFrame(
        {
            "timely": ["No"],
            "company_response": ["Closed with explanation"],
            "product": ["Debt collection"],
            "issue": ["Attempts to collect debt not owed"],
            "sub_issue": ["Debt is not yours"],
            "tags": ["Older American"],
            "word_count": [300],
            "urgency_term_count": [2],
            "has_money_language": [True],
        }
    )

    result = score_complaint_risk(frame)

    assert result.loc[0, "risk_score"] >= 5.0
    assert result.loc[0, "risk_level"] == "high"
