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


@dataclass(frozen=True)
class _FakeTimeSplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    split_timestamp: pd.Timestamp


@dataclass(frozen=True)
class _FakeBaselineResult:
    model_name: str
    metrics: object
    top_positive_terms: pd.DataFrame
    top_negative_terms: pd.DataFrame


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


def test_train_baselines_command_runs_and_writes_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "baseline_out"
    pd.DataFrame(
        {
            "Complaint ID": [1, 2, 3, 4],
            "Date received": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
            "Product": ["Credit card", "Mortgage", "Debt collection", "Bank account"],
            "Issue": ["Billing disputes", "Foreclosure", "Harassment", "Fees"],
            "Consumer complaint narrative": [
                "urgent fraud charges",
                "foreclosure process issue",
                "collector threats",
                "fees and account lock",
            ],
            "Timely response?": ["No", "Yes", "No", "Yes"],
        }
    ).to_csv(input_path, index=False)

    captured: dict[str, object] = {}

    def fake_split(
        frame: pd.DataFrame,
        date_column: str,
        train_fraction: float,
    ) -> _FakeTimeSplitResult:
        assert date_column == "date_received"
        assert train_fraction == 0.75
        train = frame.iloc[:3].copy()
        test = frame.iloc[2:].copy()
        train["target"] = [0, 1, 0]
        test["target"] = [1, 0]
        return _FakeTimeSplitResult(
            train=train,
            test=test,
            split_timestamp=pd.Timestamp("2024-03-01"),
        )

    def fake_run_baselines(**_: object) -> list[_FakeBaselineResult]:
        return [
            _FakeBaselineResult(
                model_name="logistic_regression",
                metrics=object(),
                top_positive_terms=pd.DataFrame({"term": ["fraud"], "weight": [1.1]}),
                top_negative_terms=pd.DataFrame({"term": ["resolved"], "weight": [-0.8]}),
            ),
            _FakeBaselineResult(
                model_name="linear_svm",
                metrics=object(),
                top_positive_terms=pd.DataFrame({"term": ["threat"], "weight": [0.9]}),
                top_negative_terms=pd.DataFrame({"term": ["question"], "weight": [-0.7]}),
            ),
        ]

    def fake_metrics_table(results: list[_FakeBaselineResult]) -> pd.DataFrame:
        assert len(results) == 2
        return pd.DataFrame(
            {
                "model_name": ["logistic_regression", "linear_svm"],
                "precision": [0.8, 0.7],
                "recall": [0.6, 0.65],
                "f1": [0.685, 0.674],
                "roc_auc": [0.82, 0.79],
            }
        )

    def fake_write_run_manifest(manifest: object, output_path: str | Path) -> Path:
        captured["manifest"] = manifest
        captured["manifest_output_path"] = Path(output_path)
        return Path(output_path)

    monkeypatch.setattr(cli, "time_aware_train_test_split", fake_split)
    monkeypatch.setattr(cli, "run_text_baseline_models", fake_run_baselines)
    monkeypatch.setattr(cli, "baseline_metrics_table", fake_metrics_table)
    monkeypatch.setattr(cli, "write_run_manifest", fake_write_run_manifest)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "train-baselines",
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--train-fraction",
            "0.75",
            "--min-df",
            "1",
            "--max-features",
            "200",
        ],
    )

    assert result.exit_code == 0
    assert "Baseline model outputs written" in result.output
    assert (output_dir / "baseline_metrics.csv").exists()
    assert (output_dir / "train_class_imbalance.csv").exists()
    assert (output_dir / "time_split_summary.csv").exists()
    assert (output_dir / "logistic_regression_top_positive_terms.csv").exists()
    manifest = captured["manifest"]
    assert isinstance(manifest, RunManifest)
    assert manifest.command == "train-baselines"
    assert manifest.parameters["target_definition"] == "risk_level == 'high'"
    assert captured["manifest_output_path"] == output_dir / "run_manifest.json"
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
