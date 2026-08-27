from __future__ import annotations

from localstub.http.connection import should_close_connection
from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest, RecordedHTTPRequest


def _request(
    http_version: str = "1.1",
    headers: dict[str, str] | None = None,
) -> RecordedHTTPRequest:
    request = HTTPRequest(
        method="GET",
        target="/",
        headers=Headers.from_items((headers or {}).items()),
    )
    return RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=b"",
        http_version=http_version,
    )


def test_should_close_connection_http10_response_without_keep_alive():
    assert should_close_connection(
        _request(),
        response_headers={},
        response_version="1.0",
    )


def test_should_close_connection_http10_response_with_keep_alive_persists():
    assert not should_close_connection(
        _request(),
        response_headers={"Connection": "keep-alive"},
        response_version="1.0",
    )


def test_should_close_connection_http10_response_honors_request_close():
    assert should_close_connection(
        _request(headers={"Connection": "close"}),
        response_headers={"Connection": "keep-alive"},
        response_version="1.0",
    )


def test_should_close_connection_http11_response_version_persists():
    assert not should_close_connection(
        _request(),
        response_headers={},
        response_version="1.1",
    )


def test_should_close_connection_unknown_response_version_persists():
    assert not should_close_connection(
        _request(),
        response_headers={},
    )
