from __future__ import annotations

import pandas as pd

from cx_complaints.data.validation import validate_complaints_frame


def test_validate_complaints_frame_passes_on_good_data() -> None:
    frame = pd.DataFrame(
        {
            "complaint_id": [1, 2],
            "date_received": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "product": ["Credit card", "Mortgage"],
            "sub_product": [pd.NA, pd.NA],
            "issue": ["Billing", "Foreclosure"],
            "sub_issue": [pd.NA, pd.NA],
            "complaint_what_happened": ["Text", "Text"],
            "company_public_response": [pd.NA, pd.NA],
            "company": ["A", "B"],
            "state": ["CA", "NY"],
            "zip_code": [pd.NA, pd.NA],
            "tags": [pd.NA, pd.NA],
            "consumer_consent_provided": [pd.NA, pd.NA],
            "submitted_via": [pd.NA, pd.NA],
            "date_sent_to_company": [pd.NaT, pd.NaT],
            "company_response": ["Closed", "Closed"],
            "timely": ["Yes", "Yes"],
            "consumer_disputed": [pd.NA, pd.NA],
            "has_narrative": [True, True],
        }
    )

    report = validate_complaints_frame(frame, min_narrative_share=0.5, max_missing_date_share=0.1)

    assert not report.has_failures
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["canonical_columns_present"] == "pass"
    assert statuses["missing_date_received"] == "pass"


def test_validate_complaints_frame_fails_on_missing_dates() -> None:
    frame = pd.DataFrame(
        {
            "complaint_id": [1, 2],
            "date_received": [pd.NaT, pd.NaT],
            "product": ["Credit card", "Mortgage"],
            "sub_product": [pd.NA, pd.NA],
            "issue": ["Billing", "Foreclosure"],
            "sub_issue": [pd.NA, pd.NA],
            "complaint_what_happened": ["Text", "Text"],
            "company_public_response": [pd.NA, pd.NA],
            "company": ["A", "B"],
            "state": ["CA", "NY"],
            "zip_code": [pd.NA, pd.NA],
            "tags": [pd.NA, pd.NA],
            "consumer_consent_provided": [pd.NA, pd.NA],
            "submitted_via": [pd.NA, pd.NA],
            "date_sent_to_company": [pd.NaT, pd.NaT],
            "company_response": ["Closed", "Closed"],
            "timely": ["Yes", "Yes"],
            "consumer_disputed": [pd.NA, pd.NA],
            "has_narrative": [False, False],
        }
    )

    report = validate_complaints_frame(frame, min_narrative_share=0.5, max_missing_date_share=0.1)

    assert report.has_failures
    statuses = {check.name: check.status for check in report.checks}
    assert statuses["missing_date_received"] == "fail"
