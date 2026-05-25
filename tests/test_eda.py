from __future__ import annotations

from pathlib import Path

import pandas as pd

from cx_complaints.reports.eda import write_eda_charts


def test_write_eda_charts_writes_expected_png_files(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "product": ["Credit card", "Mortgage", "Credit card"],
            "issue": ["Billing", "Foreclosure", "Billing"],
            "date_received": ["2024-01-10", "2024-02-11", "2024-02-20"],
            "risk_level": ["high", "low", "medium"],
            "timely": ["Yes", "No", "Yes"],
        }
    )
    output_dir = tmp_path / "eda"

    write_eda_charts(frame=frame, output_dir=output_dir, top_n=5)

    expected = {
        "product_share.png",
        "issue_share.png",
        "monthly_volume_trend.png",
        "risk_distribution.png",
        "timely_response_share.png",
    }
    written = {path.name for path in output_dir.iterdir()}
    assert expected == written
