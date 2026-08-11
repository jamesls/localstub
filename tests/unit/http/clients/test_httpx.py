from __future__ import annotations

import gzip
from collections.abc import AsyncIterator

import httpx
import pytest

from localstub.http.client import HTTPClientError
from localstub.http.clients.httpx import HttpxClient
from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest


class AsyncResponseStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes = b"") -> None:
        self._data = data

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self._data


def _request(uri: str) -> HTTPRequest:
    return HTTPRequest(method="GET", target=uri)


@pytest.mark.asyncio
async def test_sends_request_fields_to_backend() -> None:
    received_request: httpx.Request | None = None

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal received_request
        received_request = request
        return httpx.Response(200, stream=AsyncResponseStream())

    body = b"payload"
    request = HTTPRequest(
        method="POST",
        target="http://example.com/path",
        headers=Headers.from_items([("X-End-To-End", "preserved")]),
        body=body,
    )
    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(transport=transport) as backend:
        response = await HttpxClient(backend).send(request)

    assert response.status == 200
    assert received_request is not None
    assert received_request.method == "POST"
    assert str(received_request.url) == "http://example.com/path"
    assert received_request.content == body
    assert received_request.headers["X-End-To-End"] == "preserved"
    assert received_request.headers["Content-Length"] == str(len(body))


@pytest.mark.asyncio
async def test_bodyless_post_sends_no_framing_headers() -> None:
    received_request: httpx.Request | None = None

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal received_request
        received_request = request
        return httpx.Response(200, stream=AsyncResponseStream())

    transport = httpx.MockTransport(handle_request)
    request = HTTPRequest(method="POST", target="http://example.com/path")

    async with httpx.AsyncClient(transport=transport) as backend:
        response = await HttpxClient(backend).send(request)

    assert response.status == 200
    assert received_request is not None
    assert "Content-Length" not in received_request.headers
    assert "Transfer-Encoding" not in received_request.headers


@pytest.mark.asyncio
async def test_empty_body_sends_content_length_zero() -> None:
    received_request: httpx.Request | None = None

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal received_request
        received_request = request
        return httpx.Response(200, stream=AsyncResponseStream())

    transport = httpx.MockTransport(handle_request)
    request = HTTPRequest(
        method="GET",
        target="http://example.com/path",
        body=b"",
    )

    async with httpx.AsyncClient(transport=transport) as backend:
        response = await HttpxClient(backend).send(request)

    assert response.status == 200
    assert received_request is not None
    assert received_request.headers["Content-Length"] == "0"


@pytest.mark.asyncio
async def test_preserves_duplicate_response_headers() -> None:
    def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers=[("Set-Cookie", "a=1"), ("Set-Cookie", "b=2")],
            stream=AsyncResponseStream(b"body"),
        )

    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(transport=transport) as backend:
        response = await HttpxClient(backend).send(
            _request("http://example.com/path")
        )

    assert response.body == b"body"
    assert response.headers.get_all("Set-Cookie") == ["a=1", "b=2"]


@pytest.mark.asyncio
async def test_maps_request_error_to_client_error() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(transport=transport) as backend:
        with pytest.raises(HTTPClientError) as exc_info:
            await HttpxClient(backend).send(
                _request("http://example.com/path")
            )

    assert "connection refused" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


@pytest.mark.asyncio
async def test_hook_consumed_stream_returns_decoded_body() -> None:
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

    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(
        transport=transport,
        event_hooks={"response": [read_body_hook]},
    ) as backend:
        response = await HttpxClient(backend).send(
            _request("http://example.com/path")
        )

    assert response.status == 200
    assert response.body == body
    names = {name.lower() for name, _ in response.headers.items()}
    assert "content-encoding" not in names
    assert "content-length" not in names


@pytest.mark.asyncio
async def test_forwards_response_built_from_content_bytes() -> None:
    def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"mock body")

    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(transport=transport) as backend:
        response = await HttpxClient(backend).send(
            _request("http://example.com/path")
        )

    assert response.status == 200
    assert response.body == b"mock body"


@pytest.mark.asyncio
async def test_does_not_close_injected_backend() -> None:
    def handle_request(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=AsyncResponseStream())

    transport = httpx.MockTransport(handle_request)

    async with httpx.AsyncClient(transport=transport) as backend:
        client = HttpxClient(backend)
        await client.send(_request("http://example.com/path"))
        second = await client.send(_request("http://example.com/path"))

        assert second.status == 200
        assert not backend.is_closed
