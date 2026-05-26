"""DuckDB storage utilities for complaint analytics tables."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import duckdb
import pandas as pd


class DuckDBStore:
    """Simple DuckDB-backed store for complaint analytics data."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def write_dataframe(
        self,
        frame: pd.DataFrame,
        table_name: str,
        if_exists: Literal["replace", "append"] = "replace",
    ) -> None:
        """Write a DataFrame to a DuckDB table."""
        if if_exists not in {"replace", "append"}:
            msg = "if_exists must be either 'replace' or 'append'."
            raise ValueError(msg)

        connection = duckdb.connect(str(self.database_path))
        try:
            connection.register("input_df", frame)
            if if_exists == "replace":
                connection.execute(
                    f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM input_df"
                )
            else:
                connection.execute(f"INSERT INTO {table_name} SELECT * FROM input_df")
        finally:
            connection.close()

    def table_exists(self, table_name: str) -> bool:
        """Return whether a table exists in the DuckDB database."""
        connection = duckdb.connect(str(self.database_path))
        try:
            result = connection.execute(
                "SELECT count(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name],
            ).fetchone()
        finally:
            connection.close()
        return bool(result and result[0] > 0)

    def read_table(self, table_name: str) -> pd.DataFrame:
        """Read a full table into a pandas DataFrame."""
        connection = duckdb.connect(str(self.database_path))
        try:
            frame = connection.execute(f"SELECT * FROM {table_name}").fetch_df()
        finally:
            connection.close()
        return frame

    def query(self, sql: str) -> pd.DataFrame:
        """Run a SQL query and return the result as a DataFrame."""
        connection = duckdb.connect(str(self.database_path))
        try:
            frame = connection.execute(sql).fetch_df()
        finally:
            connection.close()
        return frame
