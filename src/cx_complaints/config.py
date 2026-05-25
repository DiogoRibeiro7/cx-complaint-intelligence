"""Project configuration objects."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic import BaseModel, Field, HttpUrl, field_validator


class ProjectConfig(BaseModel):
    """Configuration for local project paths and API access."""

    base_url: HttpUrl = Field(
        default=cast(
            HttpUrl,
            "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/",
        ),
        description="Base URL for the CFPB Consumer Complaint Database API.",
    )
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    reports_dir: Path = Path("reports")
    request_timeout_seconds: int = Field(default=60, ge=1)
    max_retries: int = Field(default=5, ge=0)
    retry_backoff_seconds: float = Field(default=1.0, ge=0)

    @field_validator("raw_data_dir", "processed_data_dir", "reports_dir")
    @classmethod
    def ensure_relative_or_absolute_path(cls, value: Path) -> Path:
        """Ensure path-like settings are real Path objects."""
        if not isinstance(value, Path):
            msg = f"Expected a pathlib.Path value, received {type(value)!r}."
            raise TypeError(msg)
        return value

    def create_directories(self) -> None:
        """Create local directories used by the project."""
        for path in (self.raw_data_dir, self.processed_data_dir, self.reports_dir):
            path.mkdir(parents=True, exist_ok=True)
