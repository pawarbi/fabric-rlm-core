from __future__ import annotations

from email.message import Message
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from fabric_rlm.knowledge import KnowledgePackage, SourceProfile
from fabric_rlm.knowledge_store import SourceBinding, SourceBindingDescriptor
from fabric_rlm.onelake_knowledge_store import (
    ConcurrentWriteError,
    OneLakeKnowledgeLocation,
    OneLakeRestTransport,
    load_onelake_knowledge_package,
    save_onelake_knowledge_package,
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

    def close(self) -> None:
        pass


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


class _MemoryOneLakeOpener:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.versions: dict[str, int] = {}

    def __call__(self, request, *, timeout):
        assert timeout == 60
        parsed = urlsplit(request.full_url)
        path = unquote(parsed.path)
        query = parse_qs(parsed.query)
        method = request.method
        if method == "HEAD":
            self._require(path)
            return self._response(path)
        if method == "GET":
            self._require(path)
            return self._response(path, data=self.objects[path])
        if method == "DELETE":
            if path not in self.objects:
                raise _http_error(404)
            self.objects.pop(path)
            self.versions.pop(path)
            return _Response(status=200)
        if method == "PUT" and request.get_header("X-ms-rename-source"):
            source = unquote(request.get_header("X-ms-rename-source"))
            self._require(source)
            if request.get_header("X-ms-source-if-match") != self._etag(source):
                raise _http_error(412)
            self._check_destination_condition(request, path)
            self.objects[path] = self.objects.pop(source)
            self.versions[path] = self.versions.pop(source) + 1
            return self._response(path, status=201)
        if method == "PUT" and query.get("resource") == ["directory"]:
            if path in self.objects:
                raise _http_error(409)
            self.objects[path] = b""
            self.versions[path] = 1
            return self._response(path, status=201)
        if method == "PUT" and query.get("resource") == ["file"]:
            self._check_destination_condition(request, path)
            self.objects[path] = b""
            self.versions[path] = self.versions.get(path, 0) + 1
            return self._response(path, status=201)
        if method == "PATCH" and query.get("action") == ["append"]:
            self._require(path)
            position = int(query["position"][0])
            if position != len(self.objects[path]):
                raise _http_error(400)
            self.objects[path] += request.data or b""
            self.versions[path] += 1
            return self._response(path, status=202)
        if method == "PATCH" and query.get("action") == ["flush"]:
            self._require(path)
            if int(query["position"][0]) != len(self.objects[path]):
                raise _http_error(400)
            self.versions[path] += 1
            return self._response(path, status=200)
        raise AssertionError(f"unexpected request: {method} {request.full_url}")

    def _check_destination_condition(self, request, path: str) -> None:
        if request.get_header("If-none-match") == "*" and path in self.objects:
            raise _http_error(412)
        expected = request.get_header("If-match")
        if expected is not None and (
            path not in self.objects or expected != self._etag(path)
        ):
            raise _http_error(412)

    def _require(self, path: str) -> None:
        if path not in self.objects:
            raise _http_error(404)

    def _etag(self, path: str) -> str:
        return f'"v{self.versions[path]}"'

    def _response(
        self,
        path: str,
        *,
        status: int = 200,
        data: bytes = b"",
    ) -> _Response:
        return _Response(
            status=status,
            data=data,
            headers={
                "Content-Length": str(len(self.objects[path])),
                "ETag": self._etag(path),
            },
        )


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


def test_abfss_save_and_load_round_trip_through_rest_transport() -> None:
    opener = _MemoryOneLakeOpener()
    transport = _transport(opener)
    location = OneLakeKnowledgeLocation(
        root=ROOT,
        locator="knowledge/package.json",
    )
    package = KnowledgePackage(
        package_id="sales.knowledge.v1",
        sources=(
            SourceProfile(
                source_id="orders",
                family="delta",
                locator="delta/v1/orders",
                snapshot_fingerprint="snapshot",
                schema_fingerprint="schema",
                schema={"order_id": {"type": "integer"}},
            ),
        ),
    )
    runtime_handle = object()

    save_onelake_knowledge_package(
        location,
        package,
        transport=transport,
    )
    loaded = load_onelake_knowledge_package(
        location,
        transport=transport,
        bindings={
            "orders": SourceBinding(
                SourceBindingDescriptor(
                    source_id="orders",
                    locator="delta/v1/orders",
                ),
                runtime_handle,
            )
        },
    )

    target = (
        "/workspace-id/lakehouse-id/Files/knowledge/package.json"
    )
    assert package.fingerprint == loaded.package.fingerprint
    assert loaded.bindings["orders"] is runtime_handle
    assert ROOT.encode() not in opener.objects[target]
