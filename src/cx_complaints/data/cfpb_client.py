"""Client for the CFPB Consumer Complaint Database API."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from requests import Response, Session

from cx_complaints.config import ProjectConfig
from cx_complaints.data.schema import normalize_complaints

JsonRecord = dict[str, Any]


@dataclass(frozen=True)
class CFPBQuery:
    """Filters accepted by the CFPB complaint search endpoint.

    Only the most useful parameters are exposed here. Additional API parameters can be
    passed through ``extra_params`` when a specific analysis needs them.
    """

    limit: int = 1000
    page_size: int = 100
    has_narrative: bool | None = True
    product: str | None = None
    company: str | None = None
    state: str | None = None
    date_received_min: str | None = None
    date_received_max: str | None = None
    search_term: str | None = None
    sort: str = "created_date_desc"
    extra_params: Mapping[str, str | int | bool] | None = None

    def __post_init__(self) -> None:
        """Validate numeric pagination settings."""
        if self.limit <= 0:
            msg = "limit must be a positive integer."
            raise ValueError(msg)
        if self.page_size <= 0:
            msg = "page_size must be a positive integer."
            raise ValueError(msg)


class CFPBClient:
    """Small API client for fetching CFPB complaints.

    The CFPB API response has historically used an Elasticsearch-like shape. This
    client accepts that format but also handles list-style JSON responses defensively.
    """

    def __init__(self, config: ProjectConfig | None = None, session: Session | None = None) -> None:
        self.config = config or ProjectConfig()
        self.session = session or requests.Session()

    def fetch(self, query: CFPBQuery) -> pd.DataFrame:
        """Fetch complaints and return a normalised DataFrame.

        Parameters
        ----------
        query:
            Query filters and pagination settings.

        Returns
        -------
        pandas.DataFrame
            Normalised complaint records.
        """
        records = list(self.iter_records(query))
        frame = pd.DataFrame.from_records(records)
        return normalize_complaints(frame)

    def iter_records(self, query: CFPBQuery) -> Iterator[JsonRecord]:
        """Yield complaint records across paginated API responses."""
        fetched = 0
        offset = 0

        while fetched < query.limit:
            current_size = min(query.page_size, query.limit - fetched)
            params = self._build_params(query=query, offset=offset, size=current_size)
            response = self.session.get(
                str(self.config.base_url),
                params=params,
                timeout=self.config.request_timeout_seconds,
            )
            self._raise_for_status(response)
            page_records = self._extract_records(response.json())

            if not page_records:
                break

            for record in page_records:
                yield record
                fetched += 1
                if fetched >= query.limit:
                    break

            if len(page_records) < current_size:
                break

            offset += len(page_records)

    def _build_params(self, query: CFPBQuery, offset: int, size: int) -> dict[str, str | int | bool]:
        """Build query parameters for the API request."""
        params: dict[str, str | int | bool] = {
            "format": "json",
            "no_aggs": "true",
            "no_highlight": "true",
            "frm": offset,
            "size": size,
            "sort": query.sort,
        }

        if query.has_narrative is not None:
            params["has_narrative"] = str(query.has_narrative).lower()
        if query.product:
            params["product"] = query.product
        if query.company:
            params["company"] = query.company
        if query.state:
            params["state"] = query.state
        if query.date_received_min:
            params["date_received_min"] = query.date_received_min
        if query.date_received_max:
            params["date_received_max"] = query.date_received_max
        if query.search_term:
            params["search_term"] = query.search_term
            params["field"] = "complaint_what_happened"

        if query.extra_params:
            params.update(query.extra_params)

        return params

    @staticmethod
    def _raise_for_status(response: Response) -> None:
        """Raise a helpful error when the CFPB API request fails."""
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            msg = f"CFPB API request failed with status {response.status_code}: {response.text[:500]}"
            raise RuntimeError(msg) from exc

    @staticmethod
    def _extract_records(payload: Any) -> list[JsonRecord]:
        """Extract records from known CFPB API response shapes."""
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, dict):
            return []

        if isinstance(payload.get("hits"), dict):
            raw_hits = payload["hits"].get("hits", [])
            records: list[JsonRecord] = []
            for item in raw_hits:
                if not isinstance(item, dict):
                    continue
                source = item.get("_source")
                if isinstance(source, dict):
                    records.append(source)
                else:
                    records.append(item)
            return records

        for key in ("results", "data", "complaints"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]

        return []


def save_complaints_csv(frame: pd.DataFrame, output_path: str) -> None:
    """Save complaint records as CSV with UTF-8 encoding."""
    path = pd.io.common.stringify_path(output_path)
    frame.to_csv(path, index=False, encoding="utf-8")
