from __future__ import annotations

import pandas as pd
import pytest

from cx_complaints.data.schema import CANONICAL_COLUMNS, normalize_complaints


def test_normalize_complaints_accepts_csv_style_columns() -> None:
    raw = pd.DataFrame(
        {
            "Complaint ID": ["123"],
            "Date received": ["2024-01-01"],
            "Product": ["Credit card"],
            "Consumer complaint narrative": ["A complaint"],
            "Timely response?": ["Yes"],
        }
    )

    result = normalize_complaints(raw)

    assert tuple(result.columns) == CANONICAL_COLUMNS
    assert result.loc[0, "complaint_id"] == 123
    assert str(result.loc[0, "date_received"].date()) == "2024-01-01"
    assert result.loc[0, "product"] == "Credit card"


def test_normalize_complaints_adds_missing_columns() -> None:
    raw = pd.DataFrame({"complaint_id": [1]})

    result = normalize_complaints(raw)

    assert set(CANONICAL_COLUMNS).issubset(result.columns)
    assert result.loc[0, "complaint_id"] == 1


def test_normalize_complaints_raises_for_non_dataframe() -> None:
    with pytest.raises(TypeError, match="Expected pandas.DataFrame"):
        normalize_complaints(frame=[])


def test_normalize_complaints_coerces_dates_and_ids() -> None:
    raw = pd.DataFrame(
        {
            "Complaint ID": ["12", "not-a-number"],
            "Date received": ["2024-01-31", "not-a-date"],
            "Date sent to company": ["2024-02-01", None],
        }
    )

    result = normalize_complaints(raw)

    assert result.loc[0, "complaint_id"] == 12
    assert pd.isna(result.loc[1, "complaint_id"])
    assert str(result.loc[0, "date_received"].date()) == "2024-01-31"
    assert pd.isna(result.loc[1, "date_received"])
    assert str(result.loc[0, "date_sent_to_company"].date()) == "2024-02-01"
    assert pd.isna(result.loc[1, "date_sent_to_company"])


def test_normalize_complaints_strips_text_columns() -> None:
    raw = pd.DataFrame(
        {
            "Complaint ID": [123],
            "Product": ["  Credit card  "],
            "Issue": ["  Billing disputes"],
        }
    )

    result = normalize_complaints(raw)

    assert result.loc[0, "product"] == "Credit card"
    assert result.loc[0, "issue"] == "Billing disputes"


def test_normalize_complaints_maps_has_narrative_to_nullable_boolean() -> None:
    raw = pd.DataFrame(
        {
            "Complaint ID": [1, 2, 3, 4, 5, 6],
            "has_narrative": ["YES", "0", True, "unknown", pd.NA, "n"],
        }
    )

    result = normalize_complaints(raw)

    assert result.loc[0, "has_narrative"]
    assert not result.loc[1, "has_narrative"]
    assert result.loc[2, "has_narrative"]
    assert pd.isna(result.loc[3, "has_narrative"])
    assert pd.isna(result.loc[4, "has_narrative"])
    assert not result.loc[5, "has_narrative"]
