"""Supervised modelling utilities for complaint prediction workflows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class TimeSplitResult:
    """Container for a chronological train/test split."""

    train: pd.DataFrame
    test: pd.DataFrame
    split_timestamp: pd.Timestamp


@dataclass(frozen=True)
class BinaryMetrics:
    """Core binary classification metrics for model evaluation."""

    precision: float
    recall: float
    f1: float
    roc_auc: float | None


def time_aware_train_test_split(
    frame: pd.DataFrame,
    date_column: str,
    train_fraction: float = 0.8,
) -> TimeSplitResult:
    """Split records chronologically into train/test partitions.

    The split preserves temporal ordering and should be used when predicting
    future outcomes from historical complaint data.
    """
    if date_column not in frame.columns:
        msg = f"Missing required date column: {date_column!r}."
        raise KeyError(msg)
    if not 0 < train_fraction < 1:
        msg = "train_fraction must be strictly between 0 and 1."
        raise ValueError(msg)
    if frame.empty:
        msg = "Cannot split an empty DataFrame."
        raise ValueError(msg)

    output = frame.copy()
    output[date_column] = pd.to_datetime(output[date_column], errors="coerce")
    output = output.dropna(subset=[date_column]).sort_values(date_column)
    if output.empty:
        msg = "No valid timestamps found after parsing date column."
        raise ValueError(msg)

    split_index = max(1, int(len(output) * train_fraction))
    split_index = min(split_index, len(output) - 1)

    train = output.iloc[:split_index].copy()
    test = output.iloc[split_index:].copy()
    split_timestamp = train[date_column].max()

    return TimeSplitResult(train=train, test=test, split_timestamp=split_timestamp)


def summarize_class_imbalance(target: pd.Series) -> pd.DataFrame:
    """Return class counts and shares for a target variable."""
    if target.empty:
        msg = "Target series is empty."
        raise ValueError(msg)

    counts = target.value_counts(dropna=False)
    summary = (
        pd.DataFrame({"class_label": counts.index.astype(str), "count": counts.values})
        .assign(share_pct=lambda df: (df["count"] / df["count"].sum() * 100).round(2))
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    return summary


def evaluate_binary_classifier(
    y_true: pd.Series,
    y_pred: pd.Series,
    y_score: pd.Series | None = None,
) -> BinaryMetrics:
    """Compute binary classification metrics for complaint prediction tasks."""
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc: float | None = None
    if y_score is not None and pd.Series(y_true).nunique(dropna=True) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))

    return BinaryMetrics(precision=precision, recall=recall, f1=f1, roc_auc=roc_auc)
