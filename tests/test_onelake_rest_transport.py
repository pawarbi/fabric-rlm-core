from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest

from fabric_rlm.onelake_knowledge_store import (
    ConcurrentWriteError,
    OneLakeRestTransport,
)


ROOT = (
    "abfss://workspace-id@onelake.dfs.fabric.microsoft.com/"
    "lakehouse-id/Files"
)
SOURCE = f"{ROOT}/knowledge/.package.tmp"
DESTINATION = f"{ROOT}/knowledge/package.json"


class _Response:
    def __init__(
        self,
        *,
        status: int = 200,
        data: bytes = b"",
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self._data = data
        self.headers = Message()
        for name, value in (headers or {}).items():
            self.headers[name] = value

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def read(self, limit: int = -1) -> bytes:
        return self._data if limit < 0 else self._data[:limit]


class _Opener:
    def __init__(self, responses: list[object]) -> None:
        self.responses = responses
        self.requests = []

    def __call__(self, request, *, timeout):
        self.requests.append((request, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _http_error(status: int) -> HTTPError:
    return HTTPError(
        DESTINATION,
        status,
        "sanitized by transport",
        Message(),
        BytesIO(),
    )


def _transport(opener: _Opener) -> OneLakeRestTransport:
    return OneLakeRestTransport(
        token_provider=lambda: "test-token",
        opener=opener,
    )


def test_rest_stat_uses_bearer_auth_and_returns_size_and_etag() -> None:
    opener = _Opener(
        [_Response(headers={"Content-Length": "12", "ETag": '"etag-1"'})]
    )

    result = _transport(opener).stat(DESTINATION)

    request, timeout = opener.requests[0]
    assert request.method == "HEAD"
    assert request.full_url == (
        "https://onelake.dfs.fabric.microsoft.com/"
        "workspace-id/lakehouse-id/Files/knowledge/package.json"
    )
    assert request.get_header("Authorization") == "Bearer test-token"
    assert request.get_header("X-ms-version") == "2023-08-03"
    assert timeout == 60
    assert result is not None
    assert result.size == 12
    assert result.etag == '"etag-1"'


def test_rest_upload_creates_appends_and_flushes_file() -> None:
    opener = _Opener(
        [
            _Response(status=201),
            _Response(status=202),
            _Response(status=200),
        ]
    )

    _transport(opener).upload(DESTINATION, b"package-bytes")

    requests = [entry[0] for entry in opener.requests]
    assert [request.method for request in requests] == ["PUT", "PATCH", "PATCH"]
    assert requests[0].full_url.endswith(
        "/knowledge/package.json?resource=file"
    )
    assert requests[0].get_header("If-none-match") == "*"
    assert requests[1].full_url.endswith(
        "/knowledge/package.json?action=append&position=0"
    )
    assert requests[1].data == b"package-bytes"
    assert requests[2].full_url.endswith(
        "/knowledge/package.json?action=flush&position=13"
    )


def test_rest_rename_no_clobber_uses_source_and_destination_conditions() -> None:
    opener = _Opener([_Response(status=201)])

    _transport(opener).rename_no_clobber(
        SOURCE,
        DESTINATION,
        source_etag='"source-etag"',
    )

    request = opener.requests[0][0]
    assert request.method == "PUT"
    assert request.get_header("If-none-match") == "*"
    assert request.get_header("X-ms-source-if-match") == '"source-etag"'
    assert request.get_header("X-ms-rename-source") == (
        "/workspace-id/lakehouse-id/Files/knowledge/.package.tmp"
    )


@pytest.mark.parametrize(
    ("destination_etag", "condition_header", "condition_value"),
    [
        ('"destination-etag"', "If-match", '"destination-etag"'),
        (None, "If-none-match", "*"),
    ],
)
def test_rest_rename_overwrite_conditions_destination(
    destination_etag: str | None,
    condition_header: str,
    condition_value: str,
) -> None:
    opener = _Opener([_Response(status=201)])

    _transport(opener).rename_overwrite(
        SOURCE,
        DESTINATION,
        source_etag='"source-etag"',
        destination_etag=destination_etag,
    )

    request = opener.requests[0][0]
    assert request.get_header(condition_header) == condition_value
    assert request.get_header("X-ms-source-if-match") == '"source-etag"'


@pytest.mark.parametrize("status", [409, 412])
def test_rest_rename_maps_precondition_failures_to_concurrent_write(
    status: int,
) -> None:
    opener = _Opener([_http_error(status)])

    with pytest.raises(ConcurrentWriteError):
        _transport(opener).rename_overwrite(
            SOURCE,
            DESTINATION,
            source_etag='"source-etag"',
            destination_etag='"destination-etag"',
        )


def test_rest_read_is_bounded_by_range_and_response_limit() -> None:
    opener = _Opener([_Response(data=b"123456789")])

    result = _transport(opener).read(DESTINATION, 5)

    request = opener.requests[0][0]
    assert request.method == "GET"
    assert request.get_header("Range") == "bytes=0-4"
    assert result == b"12345"
