"""Baseline supervised models for complaint classification tasks."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from cx_complaints.modelling.supervised import BinaryMetrics, evaluate_binary_classifier


@dataclass(frozen=True)
class BaselineResult:
    """Container for a fitted baseline model and its evaluation outputs."""

    model_name: str
    metrics: BinaryMetrics
    top_positive_terms: pd.DataFrame
    top_negative_terms: pd.DataFrame


def run_text_baseline_models(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    text_column: str,
    target_column: str,
    min_df: int = 5,
    max_features: int = 5000,
) -> list[BaselineResult]:
    """Train and evaluate transparent text baselines for binary classification."""
    _validate_training_inputs(
        train_frame=train_frame,
        test_frame=test_frame,
        text_column=text_column,
        target_column=target_column,
    )

    models: list[tuple[str, Pipeline]] = [
        (
            "logistic_regression",
            Pipeline(
                [
                    ("vectorizer", TfidfVectorizer(min_df=min_df, max_features=max_features)),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=1000,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
        ),
        (
            "linear_svm",
            Pipeline(
                [
                    ("vectorizer", TfidfVectorizer(min_df=min_df, max_features=max_features)),
                    ("classifier", LinearSVC(class_weight="balanced")),
                ]
            ),
        ),
    ]

    x_train = train_frame[text_column].fillna("").astype(str)
    y_train = train_frame[target_column].astype(int)
    x_test = test_frame[text_column].fillna("").astype(str)
    y_test = test_frame[target_column].astype(int)

    results: list[BaselineResult] = []
    for model_name, pipeline in models:
        pipeline.fit(x_train, y_train)
        predicted = pd.Series(pipeline.predict(x_test), index=test_frame.index)
        score_series = _decision_or_probability_scores(
            pipeline=pipeline,
            x_test=x_test,
            index=test_frame.index,
        )
        metrics = evaluate_binary_classifier(y_true=y_test, y_pred=predicted, y_score=score_series)
        top_positive, top_negative = _extract_weighted_terms(pipeline)
        results.append(
            BaselineResult(
                model_name=model_name,
                metrics=metrics,
                top_positive_terms=top_positive,
                top_negative_terms=top_negative,
            )
        )

    return results


def baseline_metrics_table(results: list[BaselineResult]) -> pd.DataFrame:
    """Convert baseline results into a report-friendly metrics table."""
    rows = [
        {
            "model_name": result.model_name,
            "precision": round(result.metrics.precision, 4),
            "recall": round(result.metrics.recall, 4),
            "f1": round(result.metrics.f1, 4),
            "roc_auc": (
                round(result.metrics.roc_auc, 4)
                if result.metrics.roc_auc is not None
                else pd.NA
            ),
        }
        for result in results
    ]
    return pd.DataFrame(rows)


def _validate_training_inputs(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    text_column: str,
    target_column: str,
) -> None:
    for column in (text_column, target_column):
        if column not in train_frame.columns:
            msg = f"Missing required train column: {column!r}."
            raise KeyError(msg)
        if column not in test_frame.columns:
            msg = f"Missing required test column: {column!r}."
            raise KeyError(msg)
    if train_frame.empty or test_frame.empty:
        msg = "Train and test frames must both contain at least one row."
        raise ValueError(msg)


def _decision_or_probability_scores(
    pipeline: Pipeline,
    x_test: pd.Series,
    index: pd.Index,
) -> pd.Series:
    classifier = pipeline.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        proba = pipeline.predict_proba(x_test)[:, 1]
        return pd.Series(proba, index=index)
    if hasattr(classifier, "decision_function"):
        decision = pipeline.decision_function(x_test)
        return pd.Series(decision, index=index)
    return pd.Series([0.0] * len(x_test), index=index)


def _extract_weighted_terms(
    pipeline: Pipeline,
    top_n: int = 15,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectorizer = pipeline.named_steps["vectorizer"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    weights = pd.DataFrame({"term": feature_names, "weight": coefficients})
    top_positive = weights.sort_values("weight", ascending=False).head(top_n).reset_index(drop=True)
    top_negative = weights.sort_values("weight", ascending=True).head(top_n).reset_index(drop=True)
    return top_positive, top_negative
