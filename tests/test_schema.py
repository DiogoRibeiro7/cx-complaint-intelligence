from __future__ import annotations

import pandas as pd
import pytest

from cx_complaints.data.schema import (
    CANONICAL_COLUMNS,
    normalize_complaints,
    parse_date_series,
)


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


def test_parse_date_series_parses_iso_date_only() -> None:
    """``YYYY-MM-DD`` strings should go through the strict ISO fast path."""
    result = parse_date_series(pd.Series(["2024-01-31", "2024-02-15"]))

    assert str(result.iloc[0].date()) == "2024-01-31"
    assert str(result.iloc[1].date()) == "2024-02-15"
    # Output must be tz-naive so downstream .dt.to_period() does not warn.
    assert result.dt.tz is None


def test_parse_date_series_normalises_mixed_offsets_to_utc() -> None:
    """ISO timestamps with different offsets land on naive UTC wall-clocks."""
    result = parse_date_series(
        pd.Series(["2024-06-21T12:00:00-05:00", "2024-06-21T12:00:00+02:00"])
    )

    assert result.dt.tz is None
    # 12:00 in -05:00 is 17:00 UTC; 12:00 in +02:00 is 10:00 UTC.
    assert result.iloc[0] == pd.Timestamp("2024-06-21 17:00:00")
    assert result.iloc[1] == pd.Timestamp("2024-06-21 10:00:00")


def test_parse_date_series_coerces_outliers_via_fallback() -> None:
    """Non-ISO inputs land on NaT, but the parser does not raise."""
    result = parse_date_series(pd.Series(["2024-01-31", "not-a-date", "", None]))

    assert str(result.iloc[0].date()) == "2024-01-31"
    assert pd.isna(result.iloc[1])
    assert pd.isna(result.iloc[2])
    assert pd.isna(result.iloc[3])


def test_parse_date_series_accepts_us_style_dates_via_fallback() -> None:
    """``MM/DD/YYYY`` is not ISO; the flexible fallback must catch it."""
    result = parse_date_series(pd.Series(["01/31/2024"]))

    assert str(result.iloc[0].date()) == "2024-01-31"


def test_parse_date_series_output_supports_period_bucketing() -> None:
    """The schema-wide guarantee: ``.dt.to_period`` never warns about tz."""
    import warnings

    series = parse_date_series(
        pd.Series(["2024-06-21T12:00:00-05:00", "2024-06-28T03:00:00+00:00"])
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        weekly = series.dt.to_period("W").astype(str)

    assert len(weekly) == 2
