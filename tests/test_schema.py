from __future__ import annotations

import pandas as pd

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
