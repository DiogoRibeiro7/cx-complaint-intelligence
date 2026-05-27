from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from cx_complaints.reports.notebook_utils import (
    PALETTE,
    apply_house_style,
    concentration_table,
    find_project_root,
    horizontal_bar,
    kpi_table,
    load_complaints,
    load_pipeline_outputs,
    monthly_summary,
    render_environment,
    save_figure,
    save_table,
    setup_notebook,
    topic_labels,
    write_environment_snapshot,
)


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "complaint_id": [1, 2, 3, 4],
            "date_received": pd.to_datetime(
                ["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-25"]
            ),
            "product": ["Mortgage", "Credit card", "Mortgage", "Credit card"],
            "issue": ["Foreclosure", "Billing", "Foreclosure", "Billing"],
            "company": ["A", "B", "A", "B"],
            "timely": ["Yes", "No", "Yes", "No"],
            "risk_score": [6.0, 4.0, 5.5, 2.0],
            "risk_level": ["high", "medium", "high", "low"],
            "has_narrative": [True, True, False, True],
        }
    )


def _make_project(tmp_path: Path) -> Path:
    """Create a minimal directory layout that mimics the real project root."""
    (tmp_path / "pyproject.toml").write_text("[tool.poetry]\nname='fixture'\n")
    (tmp_path / "data" / "raw").mkdir(parents=True)
    (tmp_path / "reports").mkdir(parents=True)
    return tmp_path


def test_find_project_root_locates_marker(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    nested = project / "notebooks"
    nested.mkdir()

    assert find_project_root(start=nested) == project


def test_find_project_root_raises_when_marker_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_project_root(start=tmp_path)


def test_setup_notebook_creates_artefact_dir(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    ctx = setup_notebook("99", start=project)

    assert ctx.notebook_id == "99"
    assert ctx.project_root == project
    assert ctx.artefact_dir == project / "reports" / "notebook" / "99"
    assert ctx.artefact_dir.exists()


def test_apply_house_style_is_idempotent() -> None:
    apply_house_style()
    apply_house_style()
    assert plt.rcParams["axes.spines.top"] is False


def test_kpi_table_returns_expected_metrics() -> None:
    table = kpi_table(_scored_frame())

    assert list(table["metric"]) == [
        "Total complaints",
        "High-risk complaints",
        "Medium or high-risk complaints",
        "Untimely responses",
        "Complaints with narrative",
    ]
    assert int(table.loc[table["metric"] == "High-risk complaints", "value"].iloc[0]) == 2
    assert (
        float(
            table.loc[table["metric"] == "High-risk complaints", "share_of_total_pct"].iloc[0]
        )
        == 50.0
    )


def test_concentration_table_sorts_and_filters_by_min_volume() -> None:
    table = concentration_table(_scored_frame(), "product", min_volume=2)

    assert set(table["product"]) == {"Mortgage", "Credit card"}
    assert (table["complaints"] >= 2).all()

    sparse = concentration_table(_scored_frame(), "product", min_volume=10)
    assert sparse.empty


def test_monthly_summary_aggregates_by_month() -> None:
    summary = monthly_summary(_scored_frame())

    assert list(summary["month"]) == ["2024-01", "2024-02"]
    assert int(summary.loc[summary["month"] == "2024-01", "complaints"].iloc[0]) == 2


def test_topic_labels_join_top_terms() -> None:
    topic_terms = pd.DataFrame(
        {
            "topic_id": [0, 0, 0, 1, 1],
            "rank": [1, 2, 3, 1, 2],
            "term": ["payment", "late", "fee", "credit", "report"],
            "weight": [0.5, 0.4, 0.3, 0.6, 0.5],
        }
    )

    labels = topic_labels(topic_terms, top_n=2)

    assert labels.loc[labels["topic_id"] == 0, "topic_label"].iloc[0] == "payment, late"
    assert labels.loc[labels["topic_id"] == 1, "topic_label"].iloc[0] == "credit, report"


def test_horizontal_bar_annotates_values() -> None:
    fig, ax = plt.subplots()
    horizontal_bar(ax, ["Mortgage", "Credit card"], [5, 12])
    annotation_texts = {text.get_text() for text in ax.texts}
    assert annotation_texts == {"5", "12"}
    plt.close(fig)


def test_save_figure_and_save_table_persist_files(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    ctx = setup_notebook("88", start=project)

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    figure_path = save_figure(ctx, fig, "demo_line")
    plt.close(fig)

    table_path = save_table(ctx, _scored_frame(), "demo_table")

    assert figure_path.exists()
    assert table_path.exists()


def test_load_complaints_raises_when_sample_missing(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    ctx = setup_notebook("77", start=project)

    with pytest.raises(FileNotFoundError):
        load_complaints(ctx)


def test_load_pipeline_outputs_raises_when_outputs_missing(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    ctx = setup_notebook("66", start=project)

    with pytest.raises(FileNotFoundError):
        load_pipeline_outputs(ctx)


def test_render_environment_returns_expected_fields(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    ctx = setup_notebook("55", start=project)

    env = render_environment(ctx)

    assert set(env["field"]) >= {
        "python",
        "pandas",
        "numpy",
        "matplotlib",
        "scikit-learn",
        "project_root",
        "notebook_id",
    }


def test_write_environment_snapshot_creates_json(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    ctx = setup_notebook("44", start=project)

    path = write_environment_snapshot(ctx)

    assert path.exists()
    payload = path.read_text(encoding="utf-8")
    assert "python" in payload
    assert PALETTE["primary"].startswith("#")
