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

    # The CLI must also record which topic backend produced the artefacts so
    # downstream tools can distinguish NMF from embedding runs.
    method_path = output_dir / "topic_method.json"
    assert method_path.exists()
    payload = method_path.read_text(encoding="utf-8")
    assert '"method": "nmf"' in payload


def test_analyze_command_uses_embedding_topic_method_when_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--topic-method=embedding` should route through fit_embedding_topics
    and record the embedding metadata in topic_method.json + the manifest.

    The actual encoder is mocked so the test never downloads a model.
    """
    from dataclasses import dataclass as _dc

    from cx_complaints.nlp import embedding_topics

    input_path = tmp_path / "input.csv"
    output_dir = tmp_path / "out_embedding"
    pd.DataFrame(
        {
            "Complaint ID": [1, 2],
            "Date received": ["2024-01-01", "2024-01-02"],
            "Product": ["Credit card", "Mortgage"],
            "Issue": ["Billing disputes", "Foreclosure"],
            "Consumer complaint narrative": ["fees charged twice", "lost my house"],
            "Timely response?": ["Yes", "No"],
        }
    ).to_csv(input_path, index=False)

    captured: dict[str, object] = {}

    @_dc(frozen=True)
    class _FakeEmbeddingResult:
        encoder_name: str
        embeddings: object
        cluster_model: object
        topic_terms: pd.DataFrame
        assignments: pd.DataFrame

    def fake_fit_embedding_topics(
        texts: pd.Series,
        *,
        n_topics: int,
        encoder_name: str,
    ) -> _FakeEmbeddingResult:
        captured["embedding_called"] = True
        captured["encoder_name"] = encoder_name
        captured["n_topics"] = n_topics
        return _FakeEmbeddingResult(
            encoder_name=encoder_name,
            embeddings=None,
            cluster_model=None,
            assignments=pd.DataFrame(
                {"topic_id": [0, 1], "topic_score": [0.9, 0.8]},
                index=texts.index,
            ),
            topic_terms=pd.DataFrame(
                {
                    "topic_id": [0, 1],
                    "rank": [1, 1],
                    "term": ["fees", "house"],
                    "weight": [0.9, 0.8],
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

    monkeypatch.setattr(embedding_topics, "fit_embedding_topics", fake_fit_embedding_topics)
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
            "--topic-method",
            "embedding",
            "--embedding-model",
            "my-test-model",
        ],
    )

    assert result.exit_code == 0
    assert captured.get("embedding_called") is True
    assert captured["encoder_name"] == "my-test-model"
    assert captured["n_topics"] == 2

    method_path = output_dir / "topic_method.json"
    assert method_path.exists()
    payload = method_path.read_text(encoding="utf-8")
    assert '"method": "embedding"' in payload
    assert '"model": "my-test-model"' in payload

    manifest = captured["manifest"]
    assert isinstance(manifest, RunManifest)
    assert manifest.parameters["topic_method"] == "embedding"
    assert manifest.parameters["embedding_model"] == "my-test-model"


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


def test_persist_duckdb_command_writes_core_and_report_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.csv"
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    database_path = tmp_path / "analytics" / "complaints.duckdb"
    pd.DataFrame(
        {
            "Complaint ID": [1, 2],
            "Date received": ["2024-01-01", "2024-02-01"],
            "Product": ["Credit card", "Mortgage"],
            "Issue": ["Billing disputes", "Foreclosure"],
            "Consumer complaint narrative": ["urgent dispute", "foreclosure concern"],
            "Timely response?": ["No", "Yes"],
        }
    ).to_csv(input_path, index=False)
    pd.DataFrame({"product": ["Credit card"], "complaints": [2]}).to_csv(
        reports_dir / "product_summary.csv",
        index=False,
    )

    writes: list[tuple[str, str]] = []
    captured: dict[str, object] = {}

    class _FakeDuckDBStore:
        def __init__(self, database_path: Path) -> None:
            captured["database_path"] = database_path

        def write_dataframe(self, frame: pd.DataFrame, table_name: str, if_exists: str) -> None:
            writes.append((table_name, if_exists))

    def fake_write_run_manifest(manifest: object, output_path: str | Path) -> Path:
        captured["manifest"] = manifest
        captured["manifest_output_path"] = Path(output_path)
        return Path(output_path)

    monkeypatch.setattr(cli, "DuckDBStore", _FakeDuckDBStore)
    monkeypatch.setattr(cli, "write_run_manifest", fake_write_run_manifest)

    runner = CliRunner()
    result = runner.invoke(
        cli.main,
        [
            "persist-duckdb",
            "--input",
            str(input_path),
            "--database",
            str(database_path),
            "--reports-dir",
            str(reports_dir),
            "--if-exists",
            "append",
        ],
    )

    assert result.exit_code == 0
    assert "DuckDB tables written" in result.output
    assert captured["database_path"] == database_path
    assert ("complaints", "append") in writes
    assert ("product_summary", "replace") in writes
    manifest = captured["manifest"]
    assert isinstance(manifest, RunManifest)
    assert manifest.command == "persist-duckdb"
    assert captured["manifest_output_path"] == database_path.parent / "persist_duckdb_manifest.json"
    assert manifest.validation_status == "not_run"
    assert manifest.validation_failures == 0


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
