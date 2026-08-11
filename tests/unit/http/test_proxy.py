from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from localstub.http.client import HTTPClientError
from localstub.http.headers import Headers
from localstub.http.proxy import (
    build_origin_form_request,
    forward_proxy_request,
)
from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.http.responsespec import HTTPResponse


@dataclass
class StubHTTPClient:
    response: HTTPResponse = field(default_factory=HTTPResponse)
    requests: list[HTTPRequest] = field(default_factory=list)

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        self.requests.append(request)
        return self.response


@dataclass
class FailingHTTPClient:
    message: str

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        raise HTTPClientError(self.message)


def _recorded(
    request: HTTPRequest,
    *,
    wire_raw_bytes: bytes = b"",
    http_version: str = "1.1",
) -> RecordedHTTPRequest:
    return RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=wire_raw_bytes,
        http_version=http_version,
    )


def _proxy_request(uri: str) -> RecordedHTTPRequest:
    host = uri.split("://", 1)[1].split("/", 1)[0]
    return _recorded(
        HTTPRequest(
            method="GET",
            target=uri,
            headers=Headers.from_items([("Host", host)]),
        )
    )


def _host_header(wire: bytes) -> str:
    for line in wire.split(b"\r\n")[1:]:
        name, separator, value = line.partition(b":")
        if separator and name.strip().lower() == b"host":
            return value.strip().decode("ascii")
    raise AssertionError("no Host header in request")


def test_build_origin_form_request_keeps_non_default_port_for_scheme() -> None:
    request = _proxy_request("http://example.com:443/path")
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com:443"


def test_build_origin_form_request_keeps_port_80_for_https() -> None:
    request = _proxy_request("https://example.com:80/path")
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com:80"


def test_build_origin_form_request_omits_default_http_port() -> None:
    request = _proxy_request("http://example.com:80/path")
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com"


def test_build_origin_form_request_omits_default_https_port() -> None:
    request = _proxy_request("https://example.com:443/path")
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com"


def test_build_origin_form_request_adds_host_when_missing() -> None:
    request = _recorded(
        HTTPRequest(
            method="GET",
            target="http://example.com:443/path",
        )
    )
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com:443"


def test_build_origin_form_request_keeps_prefixed_http_version() -> None:
    request = _recorded(
        HTTPRequest(
            method="GET",
            target="http://example.com/path",
            headers=Headers.from_items([("Host", "example.com")]),
        ),
        http_version="HTTP/1.0",
    )
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert wire.startswith(b"GET /path HTTP/1.0\r\n")


def test_build_origin_form_request_preserves_wire_body_framing() -> None:
    chunked_body = b"4\r\nwiki\r\n0\r\n\r\n"
    wire = (
        b"POST http://example.com/upload HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n" + chunked_body
    )
    request = _recorded(
        HTTPRequest(
            method="POST",
            target="http://example.com/upload",
            headers=Headers.from_items([
                ("Host", "example.com"),
                ("Transfer-Encoding", "chunked"),
            ]),
            body=b"wiki",
        ),
        wire_raw_bytes=wire,
    )
    uri = request.target_uri

    assert uri is not None

    origin_wire = build_origin_form_request(request, uri)

    assert origin_wire.startswith(b"POST /upload HTTP/1.1\r\n")
    assert origin_wire.endswith(b"\r\n\r\n" + chunked_body)


@pytest.mark.asyncio
async def test_forward_proxy_request_rejects_non_absolute_target() -> None:
    client = StubHTTPClient()
    recorded = _recorded(HTTPRequest(method="GET", target="/path"))

    response = await forward_proxy_request(client, recorded)

    assert response.status == 400
    assert response.body == b"Bad Request: Not an absolute URI"
    assert client.requests == []


@pytest.mark.asyncio
async def test_forward_proxy_request_strips_dynamic_request_headers() -> None:
    client = StubHTTPClient()
    recorded = _recorded(
        HTTPRequest(
            method="GET",
            target="http://example.com/path",
            headers=Headers.from_items([
                ("Host", "example.com"),
                ("Connection", "X-Hop"),
                ("X-Hop", "request-only"),
                ("Proxy-Authorization", "secret"),
                ("X-End-To-End", "preserved"),
            ]),
        )
    )

    await forward_proxy_request(client, recorded)

    assert len(client.requests) == 1
    sent = client.requests[0]
    assert sent.target == "http://example.com/path"
    assert "Host" not in sent.headers
    assert "Connection" not in sent.headers
    assert "X-Hop" not in sent.headers
    assert "Proxy-Authorization" not in sent.headers
    assert sent.headers["X-End-To-End"] == "preserved"


@pytest.mark.asyncio
async def test_forward_proxy_request_drops_stale_content_length() -> None:
    client = StubHTTPClient()
    rewritten_body = b"rewritten body"
    recorded = _recorded(
        HTTPRequest(
            method="POST",
            target="http://example.com/path",
            headers=Headers.from_items([
                ("Host", "example.com"),
                ("Content-Length", "4"),
            ]),
            body=rewritten_body,
        )
    )

    await forward_proxy_request(client, recorded)

    assert len(client.requests) == 1
    sent = client.requests[0]
    assert sent.body == rewritten_body
    assert "Content-Length" not in sent.headers


@pytest.mark.asyncio
async def test_forward_proxy_request_maps_client_error_to_502() -> None:
    client = FailingHTTPClient(message="boom")
    recorded = _proxy_request("http://example.com/path")

    response = await forward_proxy_request(client, recorded)

    assert response.status == 502
    assert response.body == b"Bad Gateway: boom"


@pytest.mark.asyncio
async def test_forward_proxy_request_strips_dynamic_response_headers() -> None:
    client = StubHTTPClient(
        response=HTTPResponse(
            status=200,
            headers=Headers.from_items([
                ("Connection", "X-Hop"),
                ("X-Hop", "response-only"),
                ("Transfer-Encoding", "chunked"),
                ("X-End-To-End", "preserved"),
            ]),
            body=b"",
        )
    )
    recorded = _proxy_request("http://example.com/path")

    response = await forward_proxy_request(client, recorded)

    names = {name.lower() for name, _ in response.headers.items()}
    assert "connection" not in names
    assert "x-hop" not in names
    assert "transfer-encoding" not in names
    assert response.headers["X-End-To-End"] == "preserved"


@pytest.mark.asyncio
async def test_forward_proxy_request_preserves_duplicate_headers() -> None:
    client = StubHTTPClient(
        response=HTTPResponse(
            status=200,
            headers=Headers.from_items([
                ("Set-Cookie", "a=1"),
                ("Set-Cookie", "b=2"),
            ]),
            body=b"payload",
        )
    )
    recorded = _proxy_request("http://example.com/path")

    response = await forward_proxy_request(client, recorded)

    assert response.status == 200
    assert response.body == b"payload"
    assert response.headers.get_all("Set-Cookie") == ["a=1", "b=2"]
