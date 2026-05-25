"""EDA chart generation for stakeholder-facing complaint analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def write_eda_charts(
    frame: pd.DataFrame,
    output_dir: str | Path,
    top_n: int = 10,
) -> None:
    """Generate and save core EDA charts for complaints data."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)

    _plot_top_category_share(
        frame=frame,
        column="product",
        title="Complaint Share by Product",
        output_path=directory / "product_share.png",
        top_n=top_n,
    )
    _plot_top_category_share(
        frame=frame,
        column="issue",
        title="Complaint Share by Issue",
        output_path=directory / "issue_share.png",
        top_n=top_n,
    )
    _plot_monthly_volume_trend(
        frame=frame,
        output_path=directory / "monthly_volume_trend.png",
    )
    _plot_risk_distribution(
        frame=frame,
        output_path=directory / "risk_distribution.png",
    )
    _plot_timely_response_share(
        frame=frame,
        output_path=directory / "timely_response_share.png",
    )


def _plot_top_category_share(
    frame: pd.DataFrame,
    column: str,
    title: str,
    output_path: Path,
    top_n: int,
) -> None:
    counts = (
        frame[column]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
        .value_counts(dropna=False)
        .head(top_n)
    )
    share = (counts / counts.sum() * 100).round(1)

    fig, axis = plt.subplots(figsize=(10, 6))
    bars = axis.barh(share.index[::-1], share.values[::-1], color="#2F6690")
    axis.bar_label(bars, fmt="%.1f%%", padding=4)
    axis.set_title(title)
    axis.set_xlabel("Share (%)")
    axis.set_xlim(0, max(share.max() * 1.15, 10))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_monthly_volume_trend(frame: pd.DataFrame, output_path: Path) -> None:
    dates = pd.to_datetime(frame["date_received"], errors="coerce")
    monthly = (
        dates.dt.to_period("M")
        .dropna()
        .astype(str)
        .value_counts()
        .sort_index()
    )

    fig, axis = plt.subplots(figsize=(10, 5))
    axis.plot(monthly.index, monthly.values, marker="o", linewidth=2, color="#3A7D44")
    axis.set_title("Monthly Complaint Volume")
    axis.set_xlabel("Month")
    axis.set_ylabel("Complaint count")
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_risk_distribution(frame: pd.DataFrame, output_path: Path) -> None:
    counts = (
        frame["risk_level"]
        .fillna("unknown")
        .astype(str)
        .str.lower()
        .value_counts()
        .reindex(["high", "medium", "low", "unknown"], fill_value=0)
    )
    total = int(counts.sum())
    share = ((counts / max(total, 1)) * 100).round(1)

    fig, axis = plt.subplots(figsize=(8, 5))
    colors = ["#B22222", "#E09F3E", "#4C9F70", "#7A7A7A"]
    bars = axis.bar(share.index, share.values, color=colors)
    axis.bar_label(bars, fmt="%.1f%%", padding=4)
    axis.set_title("Risk Level Distribution")
    axis.set_xlabel("Risk level")
    axis.set_ylabel("Share (%)")
    axis.set_ylim(0, max(share.max() * 1.2, 10))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_timely_response_share(frame: pd.DataFrame, output_path: Path) -> None:
    timely = (
        frame["timely"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
        .value_counts()
    )
    share = (timely / timely.sum() * 100).round(1)

    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(share.index, share.values, color="#6C757D")
    axis.bar_label(bars, fmt="%.1f%%", padding=4)
    axis.set_title("Timely Response Share")
    axis.set_xlabel("Timely response")
    axis.set_ylabel("Share (%)")
    axis.set_ylim(0, max(share.max() * 1.2, 10))
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
