"""Shared helpers for the analysis notebooks.

These utilities keep notebooks consistent, reproducible, and short. They:

- resolve the project root from any working directory (notebook or repo root);
- load complaints and the pipeline outputs through a single canonical loader;
- apply a single matplotlib "house style" so charts are visually uniform;
- expose small helpers that format tables and persist artefacts.

The intent is that every notebook starts with::

    from cx_complaints.reports.notebook_utils import setup_notebook
    ctx = setup_notebook("01")

and gets ready-to-use paths, styled plots, and predictable behaviour without
duplicating boilerplate.
"""

from __future__ import annotations

import json
import platform
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cx_complaints.data.schema import normalize_complaints
from cx_complaints.features.text_features import add_text_features
from cx_complaints.modelling.risk import score_complaint_risk
from cx_complaints.preprocessing.text import add_clean_text_column

PROJECT_MARKERS: tuple[str, ...] = ("pyproject.toml", "AGENTS.md")

# House palette. Sequential blues for volume, warm reds for risk, neutral gray
# for context. Hex values chosen for AA-contrast on white backgrounds.
PALETTE: dict[str, str] = {
    "primary": "#1F3A5F",
    "secondary": "#4F86C6",
    "accent": "#E07A5F",
    "warning": "#C0392B",
    "positive": "#3A7D44",
    "neutral": "#6C757D",
    "muted": "#B0B7BF",
}

RISK_COLORS: dict[str, str] = {
    "low": PALETTE["positive"],
    "medium": PALETTE["accent"],
    "high": PALETTE["warning"],
    "unknown": PALETTE["muted"],
}


@dataclass(frozen=True)
class NotebookContext:
    """Paths and identifiers for a notebook run.

    Attributes
    ----------
    notebook_id:
        Two-digit identifier matching the file prefix (e.g. ``"01"``).
    project_root:
        Resolved project root directory.
    data_raw:
        ``data/raw`` directory under the project root.
    reports_root:
        ``reports`` directory under the project root.
    pipeline_outputs:
        ``reports/sample`` directory where the ``analyze`` CLI writes outputs.
    artefact_dir:
        ``reports/notebook/<notebook_id>`` directory for this notebook's exports.
    """

    notebook_id: str
    project_root: Path
    data_raw: Path
    reports_root: Path
    pipeline_outputs: Path
    artefact_dir: Path


def find_project_root(start: Path | None = None) -> Path:
    """Locate the project root by walking up from ``start``.

    Parameters
    ----------
    start:
        Directory to begin the search from. Defaults to the current working
        directory, which is the right choice for Jupyter kernels.

    Raises
    ------
    FileNotFoundError
        If no marker file (``pyproject.toml`` or ``AGENTS.md``) is found above
        ``start``.
    """
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if any((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    msg = (
        "Could not locate project root. Expected a 'pyproject.toml' or 'AGENTS.md' "
        f"above '{current}'."
    )
    raise FileNotFoundError(msg)


def apply_house_style() -> None:
    """Apply the project matplotlib style.

    Idempotent: calling it twice is harmless. Notebooks should call this exactly
    once in their setup cell (``setup_notebook`` already does).
    """
    mpl.rcParams.update(
        {
            "figure.figsize": (11, 5),
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "savefig.bbox": "tight",
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.titlepad": 12,
            "axes.labelsize": 11,
            "axes.labelweight": "regular",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#E1E4E8",
            "grid.linewidth": 0.8,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "legend.frameon": False,
            "font.family": "DejaVu Sans",
        }
    )


def setup_notebook(notebook_id: str, *, start: Path | None = None) -> NotebookContext:
    """Initialise a notebook session.

    Applies the matplotlib house style, resolves the project root, ensures the
    notebook artefact directory exists, and returns a :class:`NotebookContext`
    with canonical paths.

    Parameters
    ----------
    notebook_id:
        Two-character notebook identifier (e.g. ``"01"``).
    start:
        Optional override for the search start directory. Mostly useful in
        tests.
    """
    apply_house_style()
    pd.set_option("display.max_rows", 30)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda value: f"{value:,.2f}")

    project_root = find_project_root(start=start)
    reports_root = project_root / "reports"
    artefact_dir = reports_root / "notebook" / notebook_id
    artefact_dir.mkdir(parents=True, exist_ok=True)

    return NotebookContext(
        notebook_id=notebook_id,
        project_root=project_root,
        data_raw=project_root / "data" / "raw",
        reports_root=reports_root,
        pipeline_outputs=reports_root / "sample",
        artefact_dir=artefact_dir,
    )


def load_complaints(
    ctx: NotebookContext,
    *,
    filename: str = "cfpb_sample.csv",
    score: bool = True,
) -> pd.DataFrame:
    """Load and prepare the complaints sample.

    Returns a fully prepared DataFrame: normalised schema, cleaned narratives,
    text features, and (by default) the heuristic risk score.

    Parameters
    ----------
    ctx:
        Notebook context returned by :func:`setup_notebook`.
    filename:
        File under ``data/raw`` to load.
    score:
        When False, skip text-feature and risk-score computation. Useful when
        the caller only needs the raw schema (e.g. the QA notebook).
    """
    path = ctx.data_raw / filename
    if not path.exists():
        msg = (
            f"Expected complaint sample at '{path}'. Run "
            "`cx-complaints fetch --output data/raw/cfpb_sample.csv` first."
        )
        raise FileNotFoundError(msg)

    raw = pd.read_csv(path)
    frame = normalize_complaints(raw)
    if not score:
        return frame

    frame = add_clean_text_column(frame)
    frame = add_text_features(frame)
    return score_complaint_risk(frame)


def load_pipeline_outputs(ctx: NotebookContext) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the CLI ``analyze`` outputs (``complaint_scores`` and ``topic_terms``).

    Returns
    -------
    scores:
        Per-complaint scored frame with ``date_received`` parsed to datetime.
    topic_terms:
        Topic-term weights table.
    """
    scores_path = ctx.pipeline_outputs / "complaint_scores.csv"
    terms_path = ctx.pipeline_outputs / "topic_terms.csv"
    if not scores_path.exists() or not terms_path.exists():
        msg = (
            f"Missing pipeline outputs under '{ctx.pipeline_outputs}'. Run "
            "`cx-complaints analyze --input data/raw/cfpb_sample.csv "
            "--output-dir reports/sample` first."
        )
        raise FileNotFoundError(msg)

    scores = pd.read_csv(scores_path)
    scores["date_received"] = pd.to_datetime(scores["date_received"], errors="coerce")
    topic_terms = pd.read_csv(terms_path)
    return scores, topic_terms


def topic_labels(topic_terms: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Build a ``topic_id`` -> ``topic_label`` lookup from the topic-terms table.

    Each label is the comma-joined ``top_n`` highest-ranked terms.
    """
    if topic_terms.empty:
        return pd.DataFrame(columns=["topic_id", "topic_label"])

    labels = (
        topic_terms.sort_values(["topic_id", "rank"])
        .groupby("topic_id")["term"]
        .apply(lambda terms: ", ".join(terms.head(top_n).astype(str)))
        .rename("topic_label")
        .reset_index()
    )
    return labels


def kpi_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the standard headline KPI table for the complaints frame.

    The table is suitable for direct display in a notebook or markdown render.
    """
    total = len(frame)
    if total == 0:
        return pd.DataFrame(columns=["metric", "value", "share_of_total_pct"])

    risk_level = frame.get("risk_level", pd.Series(dtype=object))
    timely = frame.get("timely", pd.Series(dtype=object))
    has_narrative = frame.get("has_narrative", pd.Series(dtype="boolean"))

    high = int((risk_level == "high").sum())
    medium_high = int(risk_level.isin(["medium", "high"]).sum())
    untimely = int((timely.fillna("").astype(str).str.lower() != "yes").sum())
    with_narrative = int(has_narrative.fillna(False).astype(bool).sum())

    rows = [
        ("Total complaints", total),
        ("High-risk complaints", high),
        ("Medium or high-risk complaints", medium_high),
        ("Untimely responses", untimely),
        ("Complaints with narrative", with_narrative),
    ]
    table = pd.DataFrame(rows, columns=["metric", "value"])
    table["share_of_total_pct"] = (table["value"] / total * 100).round(1)
    return table


def concentration_table(
    frame: pd.DataFrame,
    column: str,
    *,
    top_n: int = 10,
    min_volume: int = 0,
) -> pd.DataFrame:
    """Aggregate complaints by ``column`` with risk context.

    Parameters
    ----------
    frame:
        Source complaints frame (must contain ``risk_score``/``risk_level``).
    column:
        Categorical column to group by. Missing values are bucketed as
        ``"Unknown"`` so they remain visible.
    top_n:
        Maximum number of rows to return.
    min_volume:
        Drop groups below this complaint count before sorting. Helpful for
        company-level comparisons where small samples produce unstable rates.
    """
    if column not in frame.columns:
        msg = f"Missing required column: {column!r}."
        raise KeyError(msg)

    bucketed = frame.assign(
        _group=frame[column].fillna("Unknown").astype(str).str.strip()
    )
    grouped = (
        bucketed.groupby("_group", dropna=False)
        .agg(
            complaints=("_group", "size"),
            avg_risk_score=("risk_score", "mean"),
            high_risk_share_pct=("risk_level", lambda s: (s == "high").mean() * 100),
        )
        .reset_index()
        .rename(columns={"_group": column})
    )
    if min_volume > 0:
        grouped = grouped[grouped["complaints"] >= min_volume]

    grouped = grouped.sort_values(
        ["complaints", "avg_risk_score"], ascending=[False, False]
    ).head(top_n)
    grouped["avg_risk_score"] = grouped["avg_risk_score"].round(2)
    grouped["high_risk_share_pct"] = grouped["high_risk_share_pct"].round(1)
    return grouped.reset_index(drop=True)


def monthly_summary(frame: pd.DataFrame, *, date_column: str = "date_received") -> pd.DataFrame:
    """Compute per-month complaint volume and risk indicators.

    Rows without a valid date are dropped silently because they cannot be placed
    on the timeline. Callers that care about that count should consult the QA
    notebook instead.
    """
    dated = frame.copy()
    dated[date_column] = pd.to_datetime(dated[date_column], errors="coerce")
    dated = dated.dropna(subset=[date_column])
    if dated.empty:
        return pd.DataFrame(
            columns=["month", "complaints", "avg_risk_score", "high_risk_share_pct"]
        )

    dated["month"] = dated[date_column].dt.to_period("M").astype(str)
    summary = (
        dated.groupby("month")
        .agg(
            complaints=("month", "size"),
            avg_risk_score=("risk_score", "mean"),
            high_risk_share_pct=("risk_level", lambda s: (s == "high").mean() * 100),
        )
        .reset_index()
    )
    summary["avg_risk_score"] = summary["avg_risk_score"].round(2)
    summary["high_risk_share_pct"] = summary["high_risk_share_pct"].round(1)
    return summary


def horizontal_bar(
    ax: Axes,
    labels: Iterable[str],
    values: Iterable[float],
    *,
    color: str = PALETTE["primary"],
    value_format: str = "{:,.0f}",
) -> None:
    """Draw a horizontal bar chart with value labels on each bar.

    The labels are sorted so the largest value appears at the top, which is
    standard for a "top-N" chart and avoids the manual ``[::-1]`` slicing the
    notebooks were previously doing.
    """
    labels_list = list(labels)
    values_list = [float(value) for value in values]
    order = np.argsort(values_list)
    ordered_labels = [labels_list[idx] for idx in order]
    ordered_values = [values_list[idx] for idx in order]

    bars = ax.barh(ordered_labels, ordered_values, color=color)
    ax.set_axisbelow(True)
    span = max(ordered_values) if ordered_values else 0.0
    pad = span * 0.01 if span else 0.0
    for bar, value in zip(bars, ordered_values, strict=True):
        ax.annotate(
            value_format.format(value),
            xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(pad + span * 0.005, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=9,
            color=PALETTE["primary"],
        )
    ax.margins(x=0.12)


def save_figure(ctx: NotebookContext, fig: Figure, name: str) -> Path:
    """Persist a figure under the notebook artefact directory and return its path."""
    target = ctx.artefact_dir / f"{name}.png"
    fig.savefig(target)
    return target


def save_table(ctx: NotebookContext, frame: pd.DataFrame, name: str) -> Path:
    """Persist a table to ``<artefact_dir>/<name>.csv`` and return its path."""
    target = ctx.artefact_dir / f"{name}.csv"
    frame.to_csv(target, index=False)
    return target


def render_environment(ctx: NotebookContext) -> pd.DataFrame:
    """Return a small reproducibility record (Python version, key package pins).

    The notebook displays this so a reader can reproduce the run if the result
    later disagrees with theirs.
    """
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": mpl.__version__,
    }
    try:
        import sklearn

        versions["scikit-learn"] = sklearn.__version__
    except ImportError:  # pragma: no cover - sklearn is a hard dependency
        versions["scikit-learn"] = "missing"

    return pd.DataFrame(
        {
            "field": list(versions) + ["project_root", "notebook_id"],
            "value": list(versions.values())
            + [str(ctx.project_root), ctx.notebook_id],
        }
    )


def write_environment_snapshot(ctx: NotebookContext) -> Path:
    """Persist the environment snapshot as JSON for reproducibility audits."""
    snapshot: dict[str, Any] = {
        row["field"]: row["value"] for _, row in render_environment(ctx).iterrows()
    }
    target = ctx.artefact_dir / "environment.json"
    target.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
    return target
