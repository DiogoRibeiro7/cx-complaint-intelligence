"""Helpers for pipeline run metadata generation and persistence."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cx_complaints.pipeline.manifest import ManifestCommand, RunManifest, ValidationStatus


def generate_run_id(now: datetime | None = None) -> str:
    """Generate a stable run identifier with UTC timestamp prefix."""
    current = now or datetime.now(timezone.utc)
    timestamp = current.strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}-{suffix}"


def compute_file_sha256(path: str | Path) -> str:
    """Compute SHA256 for a local file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_manifest(
    *,
    command: ManifestCommand,
    started_at: datetime,
    finished_at: datetime,
    input_path: str | Path,
    input_rows: int,
    output_rows: int | None = None,
    parameters: dict[str, str | int | float | bool | None] | None = None,
    output_paths: list[str | Path] | None = None,
    validation_status: ValidationStatus = "not_run",
    validation_failures: int = 0,
    notes: str | None = None,
    run_id: str | None = None,
    input_sha256: str | None = None,
) -> RunManifest:
    """Build a validated run manifest object from raw metadata fields."""
    resolved_input = Path(input_path)
    manifest_input_sha = input_sha256 or compute_file_sha256(resolved_input)
    manifest_output_paths = [str(Path(path)) for path in (output_paths or [])]

    return RunManifest(
        run_id=run_id or generate_run_id(now=started_at),
        command=command,
        started_at=started_at,
        finished_at=finished_at,
        input_path=str(resolved_input),
        input_sha256=manifest_input_sha,
        input_rows=input_rows,
        output_rows=output_rows,
        parameters=parameters or {},
        output_paths=manifest_output_paths,
        validation_status=validation_status,
        validation_failures=validation_failures,
        notes=notes,
    )


def write_run_manifest(manifest: RunManifest, output_path: str | Path) -> Path:
    """Write a run manifest as JSON and return the written path."""
    manifest_path = Path(output_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path
