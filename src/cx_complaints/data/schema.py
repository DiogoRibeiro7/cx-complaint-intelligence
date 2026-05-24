"""Schema normalisation for CFPB complaint records."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

CANONICAL_COLUMNS: tuple[str, ...] = (
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "complaint_what_happened",
    "company_public_response",
    "company",
    "state",
    "zip_code",
    "tags",
    "consumer_consent_provided",
    "submitted_via",
    "date_sent_to_company",
    "company_response",
    "timely",
    "consumer_disputed",
    "has_narrative",
)

COLUMN_ALIASES: Mapping[str, str] = {
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "complaint_what_happened",
    "Company public response": "company_public_response",
    "Company": "company",
    "State": "state",
    "ZIP code": "zip_code",
    "Tags": "tags",
    "Consumer consent provided?": "consumer_consent_provided",
    "Submitted via": "submitted_via",
    "Date sent to company": "date_sent_to_company",
    "Company response to consumer": "company_response",
    "Timely response?": "timely",
    "Consumer disputed?": "consumer_disputed",
    "Complaint ID": "complaint_id",
}

DATE_COLUMNS: tuple[str, ...] = ("date_received", "date_sent_to_company")
TEXT_COLUMNS: tuple[str, ...] = (
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "complaint_what_happened",
    "company_public_response",
    "company",
    "state",
    "zip_code",
    "tags",
    "consumer_consent_provided",
    "submitted_via",
    "company_response",
    "timely",
    "consumer_disputed",
)


def normalize_complaints(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise CFPB complaint data into a stable internal schema.

    The function accepts either API-style snake_case fields or CSV-style title-case
    fields from the public data export.
    """
    if not isinstance(frame, pd.DataFrame):
        msg = f"Expected pandas.DataFrame, received {type(frame)!r}."
        raise TypeError(msg)

    output = frame.rename(columns=COLUMN_ALIASES).copy()

    for column in CANONICAL_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA

    output = output.loc[:, list(CANONICAL_COLUMNS)]

    for column in DATE_COLUMNS:
        output[column] = pd.to_datetime(output[column], errors="coerce")

    for column in TEXT_COLUMNS:
        output[column] = output[column].astype("string").str.strip()

    output["complaint_id"] = pd.to_numeric(output["complaint_id"], errors="coerce").astype("Int64")

    if "has_narrative" in output.columns:
        output["has_narrative"] = output["has_narrative"].map(_to_boolean).astype("boolean")

    return output


def _to_boolean(value: object) -> bool | None:
    """Convert common CFPB boolean representations to Python booleans."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None
