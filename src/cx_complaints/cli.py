"""Command-line interface for the complaint intelligence project."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import click
import pandas as pd
from rich.console import Console

from cx_complaints.data.cfpb_client import CFPBClient, CFPBQuery, save_complaints_csv
from cx_complaints.data.duckdb_store import DuckDBStore
from cx_complaints.data.schema import normalize_complaints
from cx_complaints.data.validation import validate_complaints_frame
from cx_complaints.features.text_features import add_text_features
from cx_complaints.modelling.baselines import baseline_metrics_table, run_text_baseline_models
from cx_complaints.modelling.risk import score_complaint_risk
from cx_complaints.modelling.supervised import (
    summarize_class_imbalance,
    time_aware_train_test_split,
)
from cx_complaints.nlp.topics import fit_nmf_topics
from cx_complaints.pipeline.manifest import ValidationStatus
from cx_complaints.pipeline.metadata import build_run_manifest, write_run_manifest
from cx_complaints.preprocessing.text import add_clean_text_column
from cx_complaints.reports.eda import write_eda_charts
from cx_complaints.reports.summary import write_report_outputs

console = Console()


@click.group()
def main() -> None:
    """Customer experience complaint intelligence CLI."""


@main.command()
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--limit", default=1000, show_default=True, type=int)
@click.option("--page-size", default=100, show_default=True, type=int)
@click.option("--has-narrative", default=True, show_default=True, type=bool)
@click.option("--product", default=None, type=str)
@click.option("--company", default=None, type=str)
@click.option("--state", default=None, type=str)
@click.option(
    "--date-min",
    default=None,
    type=str,
    help="Minimum date_received, for example 2024-01-01.",
)
@click.option(
    "--date-max",
    default=None,
    type=str,
    help="Maximum date_received, for example 2024-12-31.",
)
@click.option("--search-term", default=None, type=str)
def fetch(
    output_path: Path,
    limit: int,
    page_size: int,
    has_narrative: bool,
    product: str | None,
    company: str | None,
    state: str | None,
    date_min: str | None,
    date_max: str | None,
    search_term: str | None,
) -> None:
    """Fetch complaint records from the CFPB API and save them as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    query = CFPBQuery(
        limit=limit,
        page_size=page_size,
        has_narrative=has_narrative,
        product=product,
        company=company,
        state=state,
        date_received_min=date_min,
        date_received_max=date_max,
        search_term=search_term,
    )
    client = CFPBClient()
    frame = client.fetch(query)
    save_complaints_csv(frame, str(output_path))

    console.print(f"[green]Saved {len(frame):,} complaints to {output_path}[/green]")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--topics", "n_topics", default=8, show_default=True, type=int)
@click.option("--min-df", default=5, show_default=True, type=int)
@click.option("--max-features", default=5000, show_default=True, type=int)
@click.option(
    "--topic-method",
    type=click.Choice(["nmf", "embedding"], case_sensitive=False),
    default="nmf",
    show_default=True,
    help=(
        "Topic-model backend. 'nmf' is the lightweight baseline; "
        "'embedding' uses sentence transformers + KMeans."
    ),
)
@click.option(
    "--embedding-model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    show_default=True,
    help="Sentence-transformer model name. Only used when --topic-method=embedding.",
)
def analyze(
    input_path: Path,
    output_dir: Path,
    n_topics: int,
    min_df: int,
    max_features: int,
    topic_method: str,
    embedding_model: str,
) -> None:
    """Run the full complaint intelligence analysis pipeline."""
    started_at = datetime.now(timezone.utc)
    raw = pd.read_csv(input_path)
    complaints = normalize_complaints(raw)
    complaints = add_clean_text_column(complaints)
    complaints = add_text_features(complaints)
    complaints = score_complaint_risk(complaints)

    # Topic modelling should use only records with real text. We then align the
    # assignments back to the original DataFrame index.
    text_mask = complaints["clean_text"].fillna("").astype(str).str.strip() != ""

    if topic_method.lower() == "embedding":
        from cx_complaints.nlp.embedding_topics import fit_embedding_topics

        embedding_result = fit_embedding_topics(
            complaints.loc[text_mask, "clean_text"],
            n_topics=n_topics,
            encoder_name=embedding_model,
        )
        topic_terms = embedding_result.topic_terms
        topic_assignments = embedding_result.assignments
        resolved_method = "embedding"
        resolved_model = embedding_result.encoder_name
    else:
        nmf_result = fit_nmf_topics(
            complaints.loc[text_mask, "clean_text"],
            n_topics=n_topics,
            min_df=min_df,
            max_features=max_features,
        )
        topic_terms = nmf_result.topic_terms
        topic_assignments = nmf_result.assignments
        resolved_method = "nmf"
        resolved_model = "tfidf-nmf"

    complaints["topic_id"] = pd.NA
    complaints["topic_score"] = pd.NA
    complaints.loc[text_mask, ["topic_id", "topic_score"]] = topic_assignments.to_numpy()
    complaints["topic_id"] = complaints["topic_id"].astype("Int64")
    complaints["topic_score"] = pd.to_numeric(complaints["topic_score"], errors="coerce")

    write_report_outputs(
        frame=complaints,
        topic_terms=topic_terms,
        output_dir=output_dir,
    )

    # Record which method produced these outputs so downstream tools can
    # distinguish NMF runs from embedding runs in the same directory.
    output_dir.mkdir(parents=True, exist_ok=True)
    method_path = output_dir / "topic_method.json"
    method_path.write_text(
        json.dumps(
            {
                "method": resolved_method,
                "model": resolved_model,
                "n_topics": n_topics,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    finished_at = datetime.now(timezone.utc)
    manifest = build_run_manifest(
        command="analyze",
        started_at=started_at,
        finished_at=finished_at,
        input_path=input_path,
        input_rows=len(raw),
        output_rows=len(complaints),
        parameters={
            "topics": n_topics,
            "min_df": min_df,
            "max_features": max_features,
            "topic_method": resolved_method,
            "embedding_model": resolved_model if resolved_method == "embedding" else None,
        },
        output_paths=[
            output_dir / "complaint_scores.csv",
            output_dir / "topic_terms.csv",
            output_dir / "product_summary.csv",
            output_dir / "issue_summary.csv",
            output_dir / "company_summary.csv",
            output_dir / "topic_distribution.csv",
            output_dir / "executive_summary.md",
            method_path,
        ],
    )
    write_run_manifest(manifest, output_dir / "run_manifest.json")

    console.print(
        f"[green]Analysis complete ({resolved_method}). Outputs written to {output_dir}[/green]"
    )


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--top-n", default=10, show_default=True, type=int)
def eda(input_path: Path, output_dir: Path, top_n: int) -> None:
    """Generate exploratory analysis charts from complaint records."""
    started_at = datetime.now(timezone.utc)
    frame = pd.read_csv(input_path)
    complaints = normalize_complaints(frame)

    if "risk_level" not in complaints.columns or "risk_score" not in complaints.columns:
        complaints = add_clean_text_column(complaints)
        complaints = add_text_features(complaints)
        complaints = score_complaint_risk(complaints)

    write_eda_charts(
        frame=complaints,
        output_dir=output_dir,
        top_n=top_n,
    )
    finished_at = datetime.now(timezone.utc)
    manifest = build_run_manifest(
        command="eda",
        started_at=started_at,
        finished_at=finished_at,
        input_path=input_path,
        input_rows=len(frame),
        output_rows=len(complaints),
        parameters={"top_n": top_n},
        output_paths=[
            output_dir / "product_share.png",
            output_dir / "issue_share.png",
            output_dir / "monthly_volume_trend.png",
            output_dir / "risk_distribution.png",
            output_dir / "timely_response_share.png",
        ],
    )
    write_run_manifest(manifest, output_dir / "run_manifest.json")
    console.print(f"[green]EDA charts written to {output_dir}[/green]")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--min-narrative-share", default=0.2, show_default=True, type=float)
@click.option("--max-missing-date-share", default=0.05, show_default=True, type=float)
@click.option("--fail-on-error/--no-fail-on-error", default=True, show_default=True)
def validate(
    input_path: Path,
    min_narrative_share: float,
    max_missing_date_share: float,
    fail_on_error: bool,
) -> None:
    """Validate complaint dataset quality and schema conformance."""
    started_at = datetime.now(timezone.utc)
    raw = pd.read_csv(input_path)
    complaints = normalize_complaints(raw)
    report = validate_complaints_frame(
        complaints,
        min_narrative_share=min_narrative_share,
        max_missing_date_share=max_missing_date_share,
    )

    report_frame = report.to_frame()
    console.print(report_frame.to_markdown(index=False))
    failing_checks = int((report_frame["status"] == "fail").sum())
    warning_checks = int((report_frame["status"] == "warn").sum())
    validation_status: ValidationStatus = (
        "fail" if failing_checks > 0 else "warn" if warning_checks > 0 else "pass"
    )
    finished_at = datetime.now(timezone.utc)
    validation_output_dir = Path("reports") / "validation"
    manifest = build_run_manifest(
        command="validate",
        started_at=started_at,
        finished_at=finished_at,
        input_path=input_path,
        input_rows=len(raw),
        output_rows=len(complaints),
        parameters={
            "min_narrative_share": min_narrative_share,
            "max_missing_date_share": max_missing_date_share,
            "fail_on_error": fail_on_error,
        },
        output_paths=[validation_output_dir / "run_manifest.json"],
        validation_status=validation_status,
        validation_failures=failing_checks,
    )
    write_run_manifest(manifest, validation_output_dir / "run_manifest.json")

    if fail_on_error and report.has_failures:
        raise click.ClickException("Validation failed. One or more checks returned status=fail.")

    if report.has_failures:
        console.print("[yellow]Validation completed with failures (not enforced).[/yellow]")
    else:
        console.print("[green]Validation passed with no failing checks.[/green]")


@main.command("train-baselines")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--train-fraction", default=0.8, show_default=True, type=float)
@click.option("--min-df", default=5, show_default=True, type=int)
@click.option("--max-features", default=5000, show_default=True, type=int)
def train_baselines(
    input_path: Path,
    output_dir: Path,
    train_fraction: float,
    min_df: int,
    max_features: int,
) -> None:
    """Train transparent supervised baselines with a chronological split."""
    started_at = datetime.now(timezone.utc)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(input_path)
    complaints = normalize_complaints(raw)
    complaints = add_clean_text_column(complaints)
    complaints = add_text_features(complaints)
    complaints = score_complaint_risk(complaints)
    modelling_frame = complaints.copy()
    modelling_frame["target"] = (modelling_frame["risk_level"] == "high").astype(int)

    split = time_aware_train_test_split(
        frame=modelling_frame,
        date_column="date_received",
        train_fraction=train_fraction,
    )
    if split.train["target"].nunique(dropna=True) < 2:
        msg = "Training target has a single class after splitting; cannot fit baselines."
        raise click.ClickException(msg)
    if split.test["target"].nunique(dropna=True) < 2:
        msg = "Test target has a single class after splitting; metrics are not reliable."
        raise click.ClickException(msg)

    class_balance = summarize_class_imbalance(split.train["target"])
    class_balance.to_csv(output_dir / "train_class_imbalance.csv", index=False)

    results = run_text_baseline_models(
        train_frame=split.train,
        test_frame=split.test,
        text_column="clean_text",
        target_column="target",
        min_df=min_df,
        max_features=max_features,
    )
    metrics = (
        baseline_metrics_table(results)
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
    metrics.to_csv(output_dir / "baseline_metrics.csv", index=False)

    for result in results:
        result.top_positive_terms.to_csv(
            output_dir / f"{result.model_name}_top_positive_terms.csv",
            index=False,
        )
        result.top_negative_terms.to_csv(
            output_dir / f"{result.model_name}_top_negative_terms.csv",
            index=False,
        )

    split_summary = pd.DataFrame(
        [
            {
                "split_timestamp": split.split_timestamp.isoformat(),
                "train_rows": len(split.train),
                "test_rows": len(split.test),
            }
        ]
    )
    split_summary.to_csv(output_dir / "time_split_summary.csv", index=False)

    finished_at = datetime.now(timezone.utc)
    manifest = build_run_manifest(
        command="train-baselines",
        started_at=started_at,
        finished_at=finished_at,
        input_path=input_path,
        input_rows=len(raw),
        output_rows=len(modelling_frame),
        parameters={
            "train_fraction": train_fraction,
            "min_df": min_df,
            "max_features": max_features,
            "target_definition": "risk_level == 'high'",
        },
        output_paths=[
            output_dir / "train_class_imbalance.csv",
            output_dir / "baseline_metrics.csv",
            output_dir / "time_split_summary.csv",
            output_dir / "logistic_regression_top_positive_terms.csv",
            output_dir / "logistic_regression_top_negative_terms.csv",
            output_dir / "linear_svm_top_positive_terms.csv",
            output_dir / "linear_svm_top_negative_terms.csv",
        ],
    )
    write_run_manifest(manifest, output_dir / "run_manifest.json")
    console.print(f"[green]Baseline model outputs written to {output_dir}[/green]")


@main.command("persist-duckdb")
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--database", "database_path", required=True, type=click.Path(path_type=Path))
@click.option("--reports-dir", default=None, type=click.Path(exists=True, path_type=Path))
@click.option(
    "--if-exists",
    default="replace",
    show_default=True,
    type=click.Choice(["replace", "append"]),
)
def persist_duckdb(
    input_path: Path,
    database_path: Path,
    reports_dir: Path | None,
    if_exists: str,
) -> None:
    """Persist normalized complaints and report tables into DuckDB."""
    started_at = datetime.now(timezone.utc)
    raw = pd.read_csv(input_path)
    complaints = normalize_complaints(raw)
    complaints = add_clean_text_column(complaints)
    complaints = add_text_features(complaints)
    complaints = score_complaint_risk(complaints)

    store = DuckDBStore(database_path=database_path)
    store.write_dataframe(complaints, table_name="complaints", if_exists=if_exists)  # type: ignore[arg-type]

    output_paths: list[str | Path] = [database_path]
    if reports_dir is not None:
        table_to_file = {
            "product_summary": reports_dir / "product_summary.csv",
            "issue_summary": reports_dir / "issue_summary.csv",
            "company_summary": reports_dir / "company_summary.csv",
            "topic_distribution": reports_dir / "topic_distribution.csv",
            "topic_terms": reports_dir / "topic_terms.csv",
            "baseline_metrics": reports_dir / "baseline_metrics.csv",
        }
        for table_name, path in table_to_file.items():
            if path.exists():
                frame = pd.read_csv(path)
                store.write_dataframe(frame, table_name=table_name, if_exists="replace")
                output_paths.append(path)

    finished_at = datetime.now(timezone.utc)
    manifest = build_run_manifest(
        command="persist-duckdb",
        started_at=started_at,
        finished_at=finished_at,
        input_path=input_path,
        input_rows=len(raw),
        output_rows=len(complaints),
        parameters={
            "database": str(database_path),
            "reports_dir": str(reports_dir) if reports_dir is not None else None,
            "if_exists": if_exists,
        },
        output_paths=output_paths,
    )
    manifest_path = database_path.parent / "persist_duckdb_manifest.json"
    write_run_manifest(manifest, manifest_path)
    console.print(f"[green]DuckDB tables written to {database_path}[/green]")


if __name__ == "__main__":
    main()
