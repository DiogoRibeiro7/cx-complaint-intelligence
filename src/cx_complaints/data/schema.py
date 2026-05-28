"""Schema normalisation for CFPB complaint records."""

from __future__ import annotations

import warnings
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


def parse_date_series(values: pd.Series) -> pd.Series:
    """Parse a column of date-like strings into tz-naive ``datetime64[ns]``.

    The parser uses a two-stage strategy:

    1. **Strict ISO-8601 fast path** &mdash; covers both ``YYYY-MM-DD`` and full
       ISO timestamps with offsets (the two shapes CFPB emits). Anything that
       does not match falls through with ``NaT``.
    2. **Flexible fallback** &mdash; runs ``pd.to_datetime`` without a format on
       just the rows that failed the strict path. ``utc=True`` forces a
       single timezone so mixed-offset inputs do not produce warnings.

    The result is always tz-naive (timezone is normalised to UTC and then
    dropped). This keeps period bucketing (``.dt.to_period("W")``,
    ``.dt.to_period("M")``) deterministic and warning-free downstream.
    """
    strict = pd.to_datetime(values, format="ISO8601", errors="coerce", utc=True)
    non_null_values = values.notna() & (values.astype("string").str.strip() != "")
    failed_mask = strict.isna() & non_null_values
    if failed_mask.any():
        # Suppress pandas' "Could not infer format" UserWarning &mdash; the
        # fallback is intentionally format-less because the strict path has
        # already filtered out the easy cases.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Could not infer format",
                category=UserWarning,
            )
            fallback = pd.to_datetime(values[failed_mask], errors="coerce", utc=True)
        strict.loc[failed_mask] = fallback
    # tz_localize(None) on a UTC-aware Series strips the tz attribute while
    # keeping the UTC wall-clock value &mdash; callers see naive UTC timestamps.
    return strict.dt.tz_localize(None)


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
        output[column] = parse_date_series(output[column])

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
