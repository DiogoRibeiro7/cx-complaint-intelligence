from __future__ import annotations

import pandas as pd
import pytest

from cx_complaints.modelling.baselines import baseline_metrics_table, run_text_baseline_models


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clean_text": [
                "identity theft fraud credit report",
                "mortgage escrow foreclosure issue",
                "billing dispute credit card charge",
                "debt collection harassment calls",
                "checking account fee complaint",
                "loan scam urgent threat",
                "normal account question resolved",
                "simple payment issue solved",
            ],
            "target": [1, 1, 0, 1, 0, 1, 0, 0],
        }
    )


def _test_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "clean_text": [
                "fraud threat with debt collection",
                "credit card billing confusion",
                "mortgage foreclosure concern",
                "general account information",
            ],
            "target": [1, 0, 1, 0],
        }
    )


def test_run_text_baseline_models_returns_two_results() -> None:
    results = run_text_baseline_models(
        train_frame=_train_frame(),
        test_frame=_test_frame(),
        text_column="clean_text",
        target_column="target",
        min_df=1,
        max_features=200,
    )

    assert len(results) == 2
    assert {result.model_name for result in results} == {"logistic_regression", "linear_svm"}
    for result in results:
        assert 0.0 <= result.metrics.precision <= 1.0
        assert 0.0 <= result.metrics.recall <= 1.0
        assert 0.0 <= result.metrics.f1 <= 1.0
        assert not result.top_positive_terms.empty
        assert not result.top_negative_terms.empty


def test_baseline_metrics_table_has_expected_schema() -> None:
    results = run_text_baseline_models(
        train_frame=_train_frame(),
        test_frame=_test_frame(),
        text_column="clean_text",
        target_column="target",
        min_df=1,
        max_features=200,
    )

    table = baseline_metrics_table(results)

    assert list(table.columns) == ["model_name", "precision", "recall", "f1", "roc_auc"]
    assert len(table) == 2


def test_run_text_baseline_models_validates_required_columns() -> None:
    with pytest.raises(KeyError, match="Missing required train column"):
        run_text_baseline_models(
            train_frame=pd.DataFrame({"target": [0, 1]}),
            test_frame=_test_frame(),
            text_column="clean_text",
            target_column="target",
        )
