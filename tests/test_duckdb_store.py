from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from cx_complaints.data.duckdb_store import DuckDBStore


def test_duckdb_store_write_and_read_table(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.duckdb"
    store = DuckDBStore(db_path)
    frame = pd.DataFrame({"complaint_id": [1, 2], "risk_score": [3.2, 5.1]})

    store.write_dataframe(frame, table_name="complaints", if_exists="replace")
    result = store.read_table("complaints").sort_values("complaint_id").reset_index(drop=True)

    assert result.equals(frame)


def test_duckdb_store_append_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.duckdb"
    store = DuckDBStore(db_path)
    first = pd.DataFrame({"complaint_id": [1], "risk_score": [3.0]})
    second = pd.DataFrame({"complaint_id": [2], "risk_score": [4.0]})

    store.write_dataframe(first, table_name="complaints", if_exists="replace")
    store.write_dataframe(second, table_name="complaints", if_exists="append")

    result = store.read_table("complaints")
    assert len(result) == 2
    assert sorted(result["complaint_id"].tolist()) == [1, 2]


def test_duckdb_store_table_exists_and_query(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.duckdb"
    store = DuckDBStore(db_path)
    frame = pd.DataFrame({"complaint_id": [1, 2, 3], "risk_score": [1.0, 4.0, 5.0]})

    store.write_dataframe(frame, table_name="complaints", if_exists="replace")

    assert store.table_exists("complaints")
    assert not store.table_exists("missing_table")

    result = store.query("SELECT count(*) AS rows_over_3 FROM complaints WHERE risk_score > 3")
    assert int(result.loc[0, "rows_over_3"]) == 2


def test_duckdb_store_rejects_invalid_if_exists(tmp_path: Path) -> None:
    db_path = tmp_path / "analytics.duckdb"
    store = DuckDBStore(db_path)
    frame = pd.DataFrame({"complaint_id": [1]})

    with pytest.raises(ValueError, match="if_exists must be either"):
        store.write_dataframe(frame, table_name="complaints", if_exists="invalid")  # type: ignore[arg-type]
