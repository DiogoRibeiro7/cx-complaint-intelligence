from __future__ import annotations

import pandas as pd
import pytest

from cx_complaints.dashboard.data import (
    apply_filters,
    build_executive_brief,
    priority_queue,
    resolve_topic_label,
    topic_weekly_movement,
    week_over_week,
    weekly_kpis,
)


def _complaints_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "complaint_id": [1, 2, 3, 4, 5],
            "date_received": pd.to_datetime(
                [
                    "2024-06-03",
                    "2024-06-10",
                    "2024-06-17",
                    "2024-06-24",
                    "2024-07-01",
                ]
            ),
            "product": ["Mortgage", "Mortgage", "Credit card", "Credit card", "Mortgage"],
            "company": ["A", "A", "B", "B", "C"],
            "issue": ["Foreclosure", "Billing", "Billing", "Billing", "Foreclosure"],
            "timely": ["No", "Yes", "No", "Yes", "No"],
            "company_response": [
                "Closed with explanation",
                "Closed with explanation",
                "Closed",
                "Closed",
                "In progress",
            ],
            "risk_level": ["high", "low", "high", "medium", "high"],
            "risk_score": [6.0, 1.0, 5.5, 3.0, 7.0],
        }
    )


def _scored_topic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "complaint_id": [1, 2, 3, 4, 5, 6],
            "date_received": pd.to_datetime(
                [
                    "2024-06-03",
                    "2024-06-04",
                    "2024-06-11",
                    "2024-06-12",
                    "2024-06-13",
                    "2024-06-14",
                ]
            ),
            "risk_level": ["high", "low", "high", "medium", "low", "high"],
            "risk_score": [6.0, 1.0, 5.5, 3.0, 1.0, 7.0],
            "timely": ["No", "Yes", "No", "Yes", "Yes", "No"],
            "topic_id": [0, 1, 0, 1, 1, 0],
            "company_response": [
                "Closed with explanation",
                "Closed with explanation",
                "Closed",
                "Closed",
                "Closed",
                "In progress",
            ],
        }
    )


def _topic_labels() -> pd.DataFrame:
    return pd.DataFrame(
        {"topic_id": [0, 1], "topic_label": ["fees, late", "credit, report"]}
    )


def test_apply_filters_with_no_selections_returns_input_unchanged() -> None:
    frame = _complaints_frame()
    result = apply_filters(frame)
    assert result.equals(frame)


def test_apply_filters_filters_by_product_and_risk_level() -> None:
    frame = _complaints_frame()
    result = apply_filters(frame, products=["Mortgage"], risk_levels=["high"])
    assert set(result["complaint_id"]) == {1, 5}


def test_apply_filters_filters_by_date_range() -> None:
    frame = _complaints_frame()
    start = pd.Timestamp("2024-06-15")
    end = pd.Timestamp("2024-06-30")
    result = apply_filters(frame, date_range=(start, end))
    assert set(result["complaint_id"]) == {3, 4}


def test_priority_queue_returns_high_risk_untimely_weak_response() -> None:
    queue = priority_queue(_complaints_frame())
    assert set(queue["complaint_id"]) == {1, 3, 5}
    assert list(queue["risk_score"]) == sorted(queue["risk_score"], reverse=True)


def test_priority_queue_handles_empty_input() -> None:
    empty = _complaints_frame().iloc[0:0]
    assert priority_queue(empty).empty


def test_priority_queue_normalizes_whitespace_in_string_fields() -> None:
    frame = _complaints_frame().copy()
    frame.loc[frame["complaint_id"] == 1, "timely"] = " No "
    frame.loc[frame["complaint_id"] == 1, "company_response"] = " Closed with explanation "

    queue = priority_queue(frame)
    assert 1 in set(queue["complaint_id"])


def test_weekly_kpis_aggregates_per_week() -> None:
    weekly = weekly_kpis(_scored_topic_frame())
    assert len(weekly) == 2
    assert set(weekly.columns) >= {
        "week",
        "complaints",
        "high_risk_share_pct",
        "untimely_share_pct",
        "avg_risk_score",
    }


def test_week_over_week_returns_deltas_between_last_two_weeks() -> None:
    weekly = weekly_kpis(_scored_topic_frame())
    deltas = week_over_week(weekly)

    assert set(deltas["metric"]) == {
        "complaints",
        "high_risk_share_pct",
        "untimely_share_pct",
        "avg_risk_score",
    }
    complaints_row = deltas.loc[deltas["metric"] == "complaints"].iloc[0]
    assert complaints_row["current"] == 4.0
    assert complaints_row["previous"] == 2.0
    assert complaints_row["delta"] == 2.0


def test_week_over_week_returns_empty_with_single_week() -> None:
    weekly = weekly_kpis(_scored_topic_frame().iloc[:2])
    assert week_over_week(weekly).empty


def test_topic_weekly_movement_orders_by_delta() -> None:
    movement = topic_weekly_movement(_scored_topic_frame(), _topic_labels())

    assert list(movement["topic_id"]) == [0, 1]
    assert movement.iloc[0]["delta_complaints"] >= movement.iloc[-1]["delta_complaints"]


def test_topic_weekly_movement_handles_single_week() -> None:
    single_week = _scored_topic_frame().iloc[:2].copy()
    movement = topic_weekly_movement(single_week, _topic_labels())
    assert movement.empty


def test_build_executive_brief_renders_required_sections() -> None:
    scored = _scored_topic_frame()
    weekly = weekly_kpis(scored)
    movement = topic_weekly_movement(scored, _topic_labels())

    brief = build_executive_brief(
        weekly=weekly,
        topic_move=movement,
        priority_queue_count=2,
        weekly_total=int(weekly.iloc[-1]["complaints"]),
    )

    assert brief.startswith("# Weekly Executive Brief")
    assert "## What changed" in brief
    assert "## Emerging topics" in brief
    assert "## Operational queue" in brief
    assert "Priority queue: 2 complaints" in brief


def test_build_executive_brief_requires_two_weeks() -> None:
    weekly = weekly_kpis(_scored_topic_frame().iloc[:2])
    with pytest.raises(ValueError, match="two weeks"):
        build_executive_brief(
            weekly=weekly,
            topic_move=pd.DataFrame(),
            priority_queue_count=0,
            weekly_total=0,
        )


def test_resolve_topic_label_returns_fallback_for_missing_topic() -> None:
    labels = _topic_labels()
    assert resolve_topic_label(99, labels) == "Unlabeled topic"


def test_resolve_topic_label_returns_fallback_for_blank_label() -> None:
    labels = pd.DataFrame({"topic_id": [0], "topic_label": ["   "]})
    assert resolve_topic_label(0, labels) == "Unlabeled topic"
