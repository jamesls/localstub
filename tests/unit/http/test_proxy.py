from __future__ import annotations

import gzip
from collections.abc import AsyncIterator

import httpx
import pytest

from localstub.http.headers import Headers
from localstub.http.proxy import build_origin_form_request, forward_via_httpx
from localstub.http.request import HTTPRequest


class AsyncResponseStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes = b"") -> None:
        self._data = data

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._data


def _proxy_request(uri: str) -> HTTPRequest:
    host = uri.split("://", 1)[1].split("/", 1)[0]
    return HTTPRequest(
        method="GET",
        path=uri,
        http_version="1.1",
        headers=Headers.from_items([("Host", host)]),
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
    request = HTTPRequest(
        method="GET",
        path="http://example.com:443/path",
        http_version="1.1",
    )
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com:443"


@pytest.mark.asyncio
async def test_httpx_proxy_strips_dynamic_request_headers() -> None:
    received_headers: httpx.Headers | None = None

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal received_headers
        received_headers = request.headers
        return httpx.Response(200, stream=AsyncResponseStream())

    request = HTTPRequest(
        method="GET",
        path="http://example.com/path",
        http_version="1.1",
        headers=Headers.from_items([
            ("Host", "example.com"),
            ("Connection", "X-Hop"),
            ("X-Hop", "request-only"),
            ("X-End-To-End", "preserved"),
        ]),
    )
    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(transport=transport) as client:
        await forward_via_httpx(client, request)

    assert received_headers is not None
    assert "X-Hop" not in received_headers
    assert received_headers["X-End-To-End"] == "preserved"


@pytest.mark.asyncio
async def test_httpx_proxy_regenerates_content_length_after_body_rewrite() -> (
    None
):
    received_request: httpx.Request | None = None

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal received_request
        received_request = request
        return httpx.Response(200, stream=AsyncResponseStream())

    rewritten_body = b"rewritten body"
    request = HTTPRequest(
        method="POST",
        path="http://example.com/path",
        http_version="1.1",
        headers=Headers.from_items([
            ("Host", "example.com"),
            ("Content-Length", "4"),
        ]),
        body_bytes=rewritten_body,
    )
    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(transport=transport) as client:
        response = await forward_via_httpx(client, request)

    assert response.status == 200
    assert received_request is not None
    assert received_request.content == rewritten_body
    assert received_request.headers["Content-Length"] == str(
        len(rewritten_body)
    )


@pytest.mark.asyncio
async def test_httpx_proxy_strips_dynamic_response_headers() -> None:
    def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "Connection": "X-Hop",
                "X-Hop": "response-only",
                "X-End-To-End": "preserved",
            },
            stream=AsyncResponseStream(),
        )

    request = _proxy_request("http://example.com/path")
    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(transport=transport) as client:
        response = await forward_via_httpx(client, request)

    response_headers = {
        name.lower(): value for name, value in response.headers.items()
    }
    assert "connection" not in response_headers
    assert "x-hop" not in response_headers
    assert response_headers["x-end-to-end"] == "preserved"


@pytest.mark.asyncio
async def test_httpx_proxy_with_hook_consumed_stream_returns_body() -> None:
    body = b"hook consumed response"
    compressed_body = gzip.compress(body)

    def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Encoding": "gzip"},
            stream=AsyncResponseStream(compressed_body),
        )

    async def read_body_hook(response: httpx.Response) -> None:
        await response.aread()

    request = _proxy_request("http://example.com/path")
    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(
        transport=transport,
        event_hooks={"response": [read_body_hook]},
    ) as client:
        response = await forward_via_httpx(client, request)

    assert response.status == 200
    assert response.body == body
    response_header_names = {name.lower() for name in response.headers}
    assert "content-encoding" not in response_header_names
    assert "content-length" not in response_header_names


@pytest.mark.asyncio
async def test_httpx_proxy_forwards_response_built_from_content_bytes() -> (
    None
):
    def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"mock body")

    request = _proxy_request("http://example.com/path")
    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(transport=transport) as client:
        response = await forward_via_httpx(client, request)

    assert response.status == 200
    assert response.body == b"mock body"
