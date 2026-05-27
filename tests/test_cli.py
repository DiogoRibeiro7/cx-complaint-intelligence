from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pytest
from click.testing import CliRunner

from cx_complaints import cli
from cx_complaints.pipeline.manifest import RunManifest


@dataclass(frozen=True)
class _FakeTopicResult:
    assignments: pd.DataFrame
    topic_terms: pd.DataFrame


def test_analyze_command_runs_and_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        {
            "Complaint ID": [1, 2],
            "Date received": ["2024-01-01", "2024-01-02"],
            "Product": ["Credit card", "Mortgage"],
            "Issue": ["Billing disputes", "Foreclosure"],
            "Consumer complaint narrative": ["I was charged extra fees", ""],
            "Timely response?": ["Yes", "No"],
        }
    ).to_csv(input_path, index=False)

    captured: dict[str, object] = {}

    def fake_fit_nmf_topics(
        texts: pd.Series,
        n_topics: int,
        min_df: int,
        max_features: int,
    ) -> _FakeTopicResult:
        assert len(texts) == 1
        assert n_topics == 2
        assert min_df == 1
        assert max_features == 50
        return _FakeTopicResult(
            assignments=pd.DataFrame({"topic_id": [0], "topic_score": [0.8]}),
            topic_terms=pd.DataFrame(
                {
                    "topic_id": [0],
                    "rank": [1],
                    "term": ["fees"],
                    "weight": [0.8],
                }
            ),
        )

    def fake_write_report_outputs(
        frame: pd.DataFrame,
        topic_terms: pd.DataFrame,
        output_dir: str | Path,
    ) -> None:
        captured["frame"] = frame
        captured["topic_terms"] = topic_terms
        captured["output_dir"] = Path(output_dir)

    def fake_write_run_manifest(manifest: object, output_path: str | Path) -> Path:
        captured["manifest"] = manifest
        captured["manifest_output_path"] = Path(output_path)
        return Path(output_path)

    monkeypatch.setattr(cli, "fit_nmf_topics", fake_fit_nmf_topics)
    monkeypatch.setattr(cli, "write_report_outputs", fake_write_report_outputs)
    monkeypatch.setattr(cli, "write_run_manifest", fake_write_run_manifest)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "analyze",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--topics",
            "2",
            "--min-df",
            "1",
            "--max-features",
            "50",
        ],
    )

    assert result.exit_code == 0
    assert "Analysis complete" in result.output
    assert "frame" in captured

    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert "topic_id" in frame.columns
    assert "topic_score" in frame.columns
    assert frame["topic_id"].dtype.name == "Int64"
    assert captured["output_dir"] == output_dir
    manifest = captured["manifest"]
    assert isinstance(manifest, RunManifest)
    assert manifest.command == "analyze"
    assert manifest.input_rows == 2
    assert captured["manifest_output_path"] == output_dir / "run_manifest.json"


def test_eda_command_runs_and_calls_chart_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "eda_out"
    pd.DataFrame(
        {
            "Complaint ID": [10],
            "Date received": ["2024-01-01"],
            "Product": ["Credit card"],
            "Issue": ["Billing disputes"],
            "Consumer complaint narrative": ["Need urgent help with charges"],
            "Timely response?": ["No"],
        }
    ).to_csv(input_path, index=False)

    captured: dict[str, object] = {}

    def fake_write_eda_charts(frame: pd.DataFrame, output_dir: str | Path, top_n: int = 10) -> None:
        captured["frame"] = frame
        captured["output_dir"] = Path(output_dir)
        captured["top_n"] = top_n

    def fake_write_run_manifest(manifest: object, output_path: str | Path) -> Path:
        captured["manifest"] = manifest
        captured["manifest_output_path"] = Path(output_path)
        return Path(output_path)

    monkeypatch.setattr(cli, "write_eda_charts", fake_write_eda_charts)
    monkeypatch.setattr(cli, "write_run_manifest", fake_write_run_manifest)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "eda",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--top-n",
            "7",
        ],
    )

    assert result.exit_code == 0
    assert "EDA charts written" in result.output
    assert "frame" in captured

    frame = captured["frame"]
    assert isinstance(frame, pd.DataFrame)
    assert "risk_score" in frame.columns
    assert "risk_level" in frame.columns
    assert captured["output_dir"] == output_dir
    assert captured["top_n"] == 7
    manifest = captured["manifest"]
    assert isinstance(manifest, RunManifest)
    assert manifest.command == "eda"
    assert manifest.parameters["top_n"] == 7
    assert captured["manifest_output_path"] == output_dir / "run_manifest.json"


def test_validate_command_fails_on_failing_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "Complaint ID": [1, 2],
            "Date received": ["bad-date", "also-bad"],
            "Product": ["Credit card", "Mortgage"],
            "Issue": ["Billing disputes", "Foreclosure"],
            "Timely response?": ["Yes", "No"],
            "has_narrative": ["no", "no"],
        }
    ).to_csv(input_path, index=False)

    captured: dict[str, object] = {}

    def fake_write_run_manifest(manifest: object, output_path: str | Path) -> Path:
        captured["manifest"] = manifest
        captured["manifest_output_path"] = Path(output_path)
        return Path(output_path)

    monkeypatch.setattr(cli, "write_run_manifest", fake_write_run_manifest)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "validate",
            "--input",
            str(input_path),
            "--min-narrative-share",
            "0.5",
            "--max-missing-date-share",
            "0.1",
        ],
    )

    assert result.exit_code != 0
    assert "Validation failed" in result.output
    manifest = captured["manifest"]
    assert isinstance(manifest, RunManifest)
    assert manifest.command == "validate"
    assert manifest.validation_status == "fail"
    assert manifest.validation_failures > 0
    assert captured["manifest_output_path"] == Path("reports") / "validation" / "run_manifest.json"


def test_validate_command_can_skip_failure_exit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.csv"
    pd.DataFrame(
        {
            "Complaint ID": [1, 2],
            "Date received": ["bad-date", "also-bad"],
            "Product": ["Credit card", "Mortgage"],
            "Issue": ["Billing disputes", "Foreclosure"],
            "Timely response?": ["Yes", "No"],
            "has_narrative": ["no", "no"],
        }
    ).to_csv(input_path, index=False)

    captured: dict[str, object] = {}

    def fake_write_run_manifest(manifest: object, output_path: str | Path) -> Path:
        captured["manifest"] = manifest
        captured["manifest_output_path"] = Path(output_path)
        return Path(output_path)

    monkeypatch.setattr(cli, "write_run_manifest", fake_write_run_manifest)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "validate",
            "--input",
            str(input_path),
            "--no-fail-on-error",
        ],
    )

    assert result.exit_code == 0
    assert "not enforced" in result.output
    manifest = captured["manifest"]
    assert isinstance(manifest, RunManifest)
    assert manifest.command == "validate"
    assert manifest.validation_status == "fail"
