from __future__ import annotations

import pandas as pd
import pytest

from cx_complaints.modelling.supervised import (
    evaluate_binary_classifier,
    summarize_class_imbalance,
    time_aware_train_test_split,
)


def test_time_aware_train_test_split_is_chronological() -> None:
    frame = pd.DataFrame(
        {
            "date_received": ["2024-01-03", "2024-01-01", "2024-01-04", "2024-01-02"],
            "value": [3, 1, 4, 2],
        }
    )

    result = time_aware_train_test_split(frame, date_column="date_received", train_fraction=0.5)

    assert len(result.train) == 2
    assert len(result.test) == 2
    assert result.train["date_received"].max() <= result.test["date_received"].min()
    assert str(result.split_timestamp.date()) == "2024-01-02"


def test_time_aware_train_test_split_validates_input() -> None:
    frame = pd.DataFrame({"date_received": ["2024-01-01"]})

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        time_aware_train_test_split(frame, date_column="date_received", train_fraction=1.0)

    with pytest.raises(KeyError, match="Missing required date column"):
        time_aware_train_test_split(frame, date_column="missing")


def test_summarize_class_imbalance_returns_counts_and_shares() -> None:
    target = pd.Series(["high", "low", "high", "medium"])

    result = summarize_class_imbalance(target)

    assert list(result.columns) == ["class_label", "count", "share_pct"]
    assert int(result["count"].sum()) == 4
    assert float(result["share_pct"].sum()) == 100.0


def test_evaluate_binary_classifier_with_auc() -> None:
    y_true = pd.Series([0, 1, 1, 0])
    y_pred = pd.Series([0, 1, 0, 0])
    y_score = pd.Series([0.1, 0.9, 0.4, 0.2])

    metrics = evaluate_binary_classifier(y_true=y_true, y_pred=y_pred, y_score=y_score)

    assert round(metrics.precision, 2) == 1.00
    assert round(metrics.recall, 2) == 0.50
    assert round(metrics.f1, 2) == 0.67
    assert metrics.roc_auc is not None


def test_evaluate_binary_classifier_without_auc_when_single_class() -> None:
    y_true = pd.Series([1, 1, 1])
    y_pred = pd.Series([1, 0, 1])
    y_score = pd.Series([0.9, 0.2, 0.7])

    metrics = evaluate_binary_classifier(y_true=y_true, y_pred=y_pred, y_score=y_score)

    assert metrics.roc_auc is None
