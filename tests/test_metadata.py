from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cx_complaints.pipeline.metadata import (
    build_run_manifest,
    compute_file_sha256,
    generate_run_id,
    write_run_manifest,
)


def test_generate_run_id_uses_timestamp_prefix() -> None:
    run_id = generate_run_id(datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc))
    assert run_id.startswith("20260526T120000Z-")
    assert len(run_id.split("-")[1]) == 8


def test_compute_file_sha256_returns_expected_digest(tmp_path: Path) -> None:
    file_path = tmp_path / "input.txt"
    file_path.write_text("abc", encoding="utf-8")

    digest = compute_file_sha256(file_path)

    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_build_and_write_run_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "input.csv"
    input_path.write_text("a,b\n1,2\n", encoding="utf-8")
    started_at = datetime(2026, 5, 26, 12, 0, 0, tzinfo=timezone.utc)
    finished_at = started_at + timedelta(seconds=5)

    manifest = build_run_manifest(
        command="analyze",
        started_at=started_at,
        finished_at=finished_at,
        input_path=input_path,
        input_rows=1,
        output_rows=1,
        parameters={"topics": 8},
        output_paths=[tmp_path / "report.md"],
        validation_status="pass",
        validation_failures=0,
    )
    output_path = write_run_manifest(manifest, tmp_path / "run_manifest.json")

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["command"] == "analyze"
    assert payload["input_rows"] == 1
    assert payload["validation_status"] == "pass"
