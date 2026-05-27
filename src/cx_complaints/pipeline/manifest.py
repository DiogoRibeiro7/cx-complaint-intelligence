"""Run-manifest schema for pipeline traceability."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ManifestCommand = Literal[
    "fetch",
    "analyze",
    "eda",
    "validate",
    "train-baselines",
    "persist-duckdb",
]
ValidationStatus = Literal["pass", "warn", "fail", "not_run"]


class RunManifest(BaseModel):
    """Structured metadata for one pipeline run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    command: ManifestCommand
    started_at: datetime
    finished_at: datetime
    input_path: str = Field(min_length=1)
    input_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    input_rows: int = Field(ge=0)
    output_rows: int | None = Field(default=None, ge=0)
    parameters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    output_paths: list[str] = Field(default_factory=list)
    validation_status: ValidationStatus = "not_run"
    validation_failures: int = Field(default=0, ge=0)
    notes: str | None = None

    @field_validator("output_paths")
    @classmethod
    def ensure_non_empty_output_paths(cls, value: list[str]) -> list[str]:
        """Reject blank output path entries."""
        if any(path.strip() == "" for path in value):
            msg = "output_paths cannot contain empty values."
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def validate_timestamps(self) -> RunManifest:
        """Ensure run timestamps are in chronological order."""
        if self.finished_at < self.started_at:
            msg = "finished_at cannot be earlier than started_at."
            raise ValueError(msg)
        return self

    @property
    def duration_seconds(self) -> float:
        """Return run duration in seconds."""
        return (self.finished_at - self.started_at).total_seconds()
