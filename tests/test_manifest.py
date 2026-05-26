from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pydantic import ValidationError

from cx_complaints.pipeline.manifest import RunManifest


def test_run_manifest_accepts_valid_payload() -> None:
    started = datetime(2026, 5, 26, 10, 0, 0)
    finished = started + timedelta(seconds=12)

    manifest = RunManifest(
        run_id="20260526T100000Z-abc123",
        command="analyze",
        started_at=started,
        finished_at=finished,
        input_path="data/raw/cfpb_sample.csv",
        input_sha256="a" * 64,
        input_rows=1000,
        output_rows=950,
        parameters={"topics": 8, "min_df": 5},
        output_paths=["reports/sample/executive_summary.md"],
        validation_status="pass",
        validation_failures=0,
    )

    assert manifest.duration_seconds == 12.0


def test_run_manifest_rejects_invalid_hash() -> None:
    with pytest.raises(ValidationError, match="input_sha256"):
        RunManifest(
            run_id="run-1",
            command="validate",
            started_at=datetime(2026, 5, 26, 10, 0, 0),
            finished_at=datetime(2026, 5, 26, 10, 0, 1),
            input_path="data/raw/cfpb_sample.csv",
            input_sha256="not-a-hash",
            input_rows=10,
        )


def test_run_manifest_rejects_inverted_timestamps() -> None:
    with pytest.raises(ValidationError, match="finished_at cannot be earlier"):
        RunManifest(
            run_id="run-2",
            command="eda",
            started_at=datetime(2026, 5, 26, 10, 0, 2),
            finished_at=datetime(2026, 5, 26, 10, 0, 1),
            input_path="data/raw/cfpb_sample.csv",
            input_rows=10,
        )
