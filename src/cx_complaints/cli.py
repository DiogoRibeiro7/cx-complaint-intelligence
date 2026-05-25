"""Command-line interface for the complaint intelligence project."""

from __future__ import annotations

from pathlib import Path

import click
import pandas as pd
from rich.console import Console

from cx_complaints.data.cfpb_client import CFPBClient, CFPBQuery, save_complaints_csv
from cx_complaints.data.schema import normalize_complaints
from cx_complaints.features.text_features import add_text_features
from cx_complaints.modelling.risk import score_complaint_risk
from cx_complaints.nlp.topics import fit_nmf_topics
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
def analyze(
    input_path: Path,
    output_dir: Path,
    n_topics: int,
    min_df: int,
    max_features: int,
) -> None:
    """Run the full complaint intelligence analysis pipeline."""
    raw = pd.read_csv(input_path)
    complaints = normalize_complaints(raw)
    complaints = add_clean_text_column(complaints)
    complaints = add_text_features(complaints)
    complaints = score_complaint_risk(complaints)

    # Topic modelling should use only records with real text. We then align the
    # assignments back to the original DataFrame index.
    text_mask = complaints["clean_text"].fillna("").astype(str).str.strip() != ""
    topic_result = fit_nmf_topics(
        complaints.loc[text_mask, "clean_text"],
        n_topics=n_topics,
        min_df=min_df,
        max_features=max_features,
    )

    complaints["topic_id"] = pd.NA
    complaints["topic_score"] = pd.NA
    complaints.loc[text_mask, ["topic_id", "topic_score"]] = topic_result.assignments.to_numpy()
    complaints["topic_id"] = complaints["topic_id"].astype("Int64")
    complaints["topic_score"] = pd.to_numeric(complaints["topic_score"], errors="coerce")

    write_report_outputs(
        frame=complaints,
        topic_terms=topic_result.topic_terms,
        output_dir=output_dir,
    )

    console.print(f"[green]Analysis complete. Outputs written to {output_dir}[/green]")


@main.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output-dir", required=True, type=click.Path(path_type=Path))
@click.option("--top-n", default=10, show_default=True, type=int)
def eda(input_path: Path, output_dir: Path, top_n: int) -> None:
    """Generate exploratory analysis charts from complaint records."""
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
    console.print(f"[green]EDA charts written to {output_dir}[/green]")


if __name__ == "__main__":
    main()
