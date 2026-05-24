"""Client for the CFPB Consumer Complaint Database API."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
        self.session = session or self._build_session(self.config)

    @staticmethod
    def _build_session(config: ProjectConfig) -> Session:
        """Build a requests Session with retry/backoff for transient errors."""
        session = requests.Session()
        retry = Retry(
            total=config.max_retries,
            connect=config.max_retries,
            read=config.max_retries,
            status=config.max_retries,
            backoff_factor=config.retry_backoff_seconds,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

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
            page_records = self._fetch_page(params)

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

    # Errors that can happen mid-response (after the HTTP status arrives) and
    # therefore are not retried by the urllib3 HTTPAdapter we mount above.
    _STREAMING_RETRIABLE: tuple[type[BaseException], ...] = (
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ReadTimeout,
        requests.exceptions.ConnectionError,
    )

    def _fetch_page(self, params: Mapping[str, str | int | bool]) -> list[JsonRecord]:
        """GET one page of results, retrying on transient body-read errors."""
        last_error: BaseException | None = None
        attempts = self.config.max_retries + 1
        for attempt in range(attempts):
            try:
                response = self.session.get(
                    str(self.config.base_url),
                    params=params,
                    timeout=self.config.request_timeout_seconds,
                )
                self._raise_for_status(response)
                return self._extract_records(response.json())
            except self._STREAMING_RETRIABLE as exc:
                last_error = exc
                if attempt == attempts - 1:
                    break
                sleep_for = self.config.retry_backoff_seconds * (2**attempt)
                time.sleep(sleep_for)
        msg = f"CFPB API page fetch failed after {attempts} attempts: {last_error}"
        raise RuntimeError(msg) from last_error

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
