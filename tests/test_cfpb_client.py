from __future__ import annotations

import pytest
import requests

from cx_complaints.config import ProjectConfig
from cx_complaints.data.cfpb_client import CFPBClient


class _FakeSession:
    """Session double that scripts a sequence of behaviours per call."""

    def __init__(
        self,
        behaviours: list[BaseException | _FakeResponse],
    ) -> None:
        self._behaviours = list(behaviours)
        self.calls = 0

    def get(self, *_args: object, **_kwargs: object) -> _FakeResponse:
        self.calls += 1
        if not self._behaviours:
            raise AssertionError("FakeSession ran out of scripted behaviours")
        item = self._behaviours.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _hits(records: list[dict[str, object]]) -> dict[str, object]:
    return {"hits": {"hits": [{"_source": r} for r in records]}}


def test_fetch_page_retries_on_chunked_encoding_error() -> None:
    config = ProjectConfig(max_retries=3, retry_backoff_seconds=0)
    session = _FakeSession(
        [
            requests.exceptions.ChunkedEncodingError("boom"),
            requests.exceptions.ReadTimeout("slow"),
            _FakeResponse(_hits([{"complaint_id": "1"}])),
        ]
    )
    client = CFPBClient(config=config, session=session)  # type: ignore[arg-type]

    records = client._fetch_page({})

    assert session.calls == 3
    assert records == [{"complaint_id": "1"}]


def test_extract_records_unwraps_source_in_top_level_list() -> None:
    payload = [
        {
            "_index": "complaint-public-v1",
            "_id": "9316329",
            "_source": {"complaint_id": "9316329", "product": "Credit card"},
            "sort": [57],
        },
        {"_source": {"complaint_id": "9316330", "product": "Mortgage"}},
    ]

    records = CFPBClient._extract_records(payload)

    assert records == [
        {"complaint_id": "9316329", "product": "Credit card"},
        {"complaint_id": "9316330", "product": "Mortgage"},
    ]


def test_fetch_page_gives_up_after_max_retries() -> None:
    config = ProjectConfig(max_retries=2, retry_backoff_seconds=0)
    session = _FakeSession(
        [requests.exceptions.ChunkedEncodingError("boom")] * 3
    )
    client = CFPBClient(config=config, session=session)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        client._fetch_page({})

    assert session.calls == 3
