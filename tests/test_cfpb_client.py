from __future__ import annotations

from typing import Any

import pytest
import requests

from cx_complaints.config import ProjectConfig
from cx_complaints.data.cfpb_client import CFPBClient


class _FakeSession:
    """Session double that scripts a sequence of behaviours per call."""

    def __init__(self, behaviours: list[Any]) -> None:
        self._behaviours = list(behaviours)
        self.calls = 0

    def get(self, *_args: Any, **_kwargs: Any) -> Any:
        self.calls += 1
        if not self._behaviours:
            raise AssertionError("FakeSession ran out of scripted behaviours")
        item = self._behaviours.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"status {self.status_code}")


def _hits(records: list[dict[str, Any]]) -> dict[str, Any]:
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


def test_fetch_page_gives_up_after_max_retries() -> None:
    config = ProjectConfig(max_retries=2, retry_backoff_seconds=0)
    session = _FakeSession(
        [requests.exceptions.ChunkedEncodingError("boom")] * 3
    )
    client = CFPBClient(config=config, session=session)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="failed after 3 attempts"):
        client._fetch_page({})

    assert session.calls == 3
