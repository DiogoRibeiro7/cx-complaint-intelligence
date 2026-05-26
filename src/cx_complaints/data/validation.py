"""Data validation checks for complaint analytical tables."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from cx_complaints.data.schema import CANONICAL_COLUMNS


@dataclass(frozen=True)
class ValidationCheck:
    """Result of one validation rule."""

    name: str
    status: str
    count: int
    message: str


@dataclass(frozen=True)
class ValidationReport:
    """Aggregate validation output for complaint tables."""

    checks: list[ValidationCheck]

    @property
    def has_failures(self) -> bool:
        """Return whether at least one check failed."""
        return any(check.status == "fail" for check in self.checks)

    def to_frame(self) -> pd.DataFrame:
        """Render checks as a DataFrame for CLI/reporting."""
        return pd.DataFrame(
            [
                {
                    "name": check.name,
                    "status": check.status,
                    "count": check.count,
                    "message": check.message,
                }
                for check in self.checks
            ]
        )


def validate_complaints_frame(
    frame: pd.DataFrame,
    *,
    min_narrative_share: float = 0.2,
    max_missing_date_share: float = 0.05,
) -> ValidationReport:
    """Validate key data-quality and schema checks for complaint data."""
    if not 0 <= min_narrative_share <= 1:
        msg = "min_narrative_share must be between 0 and 1."
        raise ValueError(msg)
    if not 0 <= max_missing_date_share <= 1:
        msg = "max_missing_date_share must be between 0 and 1."
        raise ValueError(msg)

    checks: list[ValidationCheck] = []

    missing_columns = [column for column in CANONICAL_COLUMNS if column not in frame.columns]
    checks.append(
        ValidationCheck(
            name="canonical_columns_present",
            status="fail" if missing_columns else "pass",
            count=len(missing_columns),
            message="Missing canonical columns: " + ", ".join(missing_columns)
            if missing_columns
            else "All canonical columns are present.",
        )
    )

    total_rows = len(frame)
    complaint_id = (
        frame["complaint_id"] if "complaint_id" in frame.columns else pd.Series(dtype="Int64")
    )
    duplicate_ids = (
        int(complaint_id.duplicated(keep=False).sum()) if not complaint_id.empty else total_rows
    )
    checks.append(
        ValidationCheck(
            name="duplicate_complaint_ids",
            status="warn" if duplicate_ids > 0 else "pass",
            count=duplicate_ids,
            message=(
                "Duplicate complaint IDs detected."
                if duplicate_ids > 0
                else "No duplicate complaint IDs."
            ),
        )
    )

    if "date_received" in frame.columns:
        missing_dates = int(frame["date_received"].isna().sum())
    else:
        missing_dates = total_rows
    missing_date_share = missing_dates / max(total_rows, 1)
    checks.append(
        ValidationCheck(
            name="missing_date_received",
            status="fail" if missing_date_share > max_missing_date_share else "pass",
            count=missing_dates,
            message=(
                f"Missing date_received share {missing_date_share:.1%} exceeds threshold "
                f"{max_missing_date_share:.1%}."
                if missing_date_share > max_missing_date_share
                else f"Missing date_received share is {missing_date_share:.1%}."
            ),
        )
    )

    if "has_narrative" in frame.columns:
        narrative_share = frame["has_narrative"].fillna(False).astype(bool).mean()
    else:
        narrative_share = 0.0
    checks.append(
        ValidationCheck(
            name="narrative_coverage",
            status="warn" if narrative_share < min_narrative_share else "pass",
            count=int(round(narrative_share * total_rows)),
            message=(
                (
                    f"Narrative share {narrative_share:.1%} is below threshold "
                    f"{min_narrative_share:.1%}."
                )
                if narrative_share < min_narrative_share
                else f"Narrative share is {narrative_share:.1%}."
            ),
        )
    )

    return ValidationReport(checks=checks)
