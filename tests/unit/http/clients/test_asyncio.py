from __future__ import annotations

import asyncio
import json
import socket
from typing import Self

import pytest

from localstub.http.client import HTTPClientError
from localstub.http.clients.asyncio import AsyncioClient
from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest
from localstub.server import AsyncHTTPTestServer


class OneShotServer:
    """Server speaking canned response bytes for one connection."""

    def __init__(self, response_bytes: bytes) -> None:
        self._response_bytes = response_bytes
        self._server: asyncio.Server | None = None
        self.port: int = 0
        self.received: bytes = b""

    async def __aenter__(self) -> Self:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object):
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = await reader.read(1024)
            if not chunk:
                break
            data += chunk
        self.received = data
        writer.write(self._response_bytes)
        await writer.drain()
        writer.close()


def _unused_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port: int = sock.getsockname()[1]
    return port


@pytest.mark.asyncio
async def test_get_exchange_against_test_server() -> None:
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"ok": True})
        client = AsyncioClient()

        response = await client.send(
            HTTPRequest(
                method="GET",
                target=f"http://{server.host}:{server.port}/items?x=1",
            )
        )

        assert response.status == 200
        assert isinstance(response.body, bytes)
        assert json.loads(response.body) == {"ok": True}
        assert server.last_request is not None
        assert server.last_request.method == "GET"
        assert server.last_request.target == "/items?x=1"
        host = server.last_request.headers["Host"]
        assert host == f"{server.host}:{server.port}"


@pytest.mark.asyncio
async def test_post_body_gets_content_length_framing() -> None:
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({})
        client = AsyncioClient()
        body = b"hello upstream"

        response = await client.send(
            HTTPRequest(
                method="POST",
                target=f"http://{server.host}:{server.port}/upload",
                headers=Headers.from_items([("X-Custom", "value")]),
                body=body,
            )
        )

        assert response.status == 200
        assert server.last_request is not None
        assert server.last_request.body == body
        content_length = server.last_request.headers["Content-Length"]
        assert content_length == str(len(body))
        assert server.last_request.headers["X-Custom"] == "value"


@pytest.mark.asyncio
async def test_parses_chunked_response_body() -> None:
    wire = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"4\r\nwiki\r\n5\r\npedia\r\n0\r\n\r\n"
    )
    async with OneShotServer(wire) as upstream:
        client = AsyncioClient()

        response = await client.send(
            HTTPRequest(
                method="GET",
                target=f"http://127.0.0.1:{upstream.port}/",
            )
        )

    assert response.status == 200
    assert response.body == b"wikipedia"


@pytest.mark.asyncio
async def test_parses_close_delimited_response_body() -> None:
    wire = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nuntil close"
    async with OneShotServer(wire) as upstream:
        client = AsyncioClient()

        response = await client.send(
            HTTPRequest(
                method="GET",
                target=f"http://127.0.0.1:{upstream.port}/",
            )
        )

    assert response.status == 200
    assert response.body == b"until close"


@pytest.mark.asyncio
async def test_head_response_with_content_length_does_not_hang() -> None:
    wire = b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\n"
    async with OneShotServer(wire) as upstream:
        client = AsyncioClient(read_timeout=5.0)

        response = await client.send(
            HTTPRequest(
                method="HEAD",
                target=f"http://127.0.0.1:{upstream.port}/",
            )
        )

    assert response.status == 200
    assert response.body == b""
    assert response.headers["Content-Length"] == "5"


@pytest.mark.asyncio
async def test_preserves_duplicate_response_headers() -> None:
    wire = (
        b"HTTP/1.1 200 OK\r\n"
        b"Set-Cookie: a=1\r\n"
        b"Set-Cookie: b=2\r\n"
        b"Content-Length: 0\r\n"
        b"\r\n"
    )
    async with OneShotServer(wire) as upstream:
        client = AsyncioClient()

        response = await client.send(
            HTTPRequest(
                method="GET",
                target=f"http://127.0.0.1:{upstream.port}/",
            )
        )

    assert response.headers.get_all("Set-Cookie") == ["a=1", "b=2"]


@pytest.mark.asyncio
async def test_skips_interim_responses_until_final() -> None:
    wire = (
        b"HTTP/1.1 100 Continue\r\n\r\n"
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 4\r\n"
        b"\r\n"
        b"done"
    )
    async with OneShotServer(wire) as upstream:
        client = AsyncioClient()

        response = await client.send(
            HTTPRequest(
                method="POST",
                target=f"http://127.0.0.1:{upstream.port}/",
                body=b"",
            )
        )

    assert response.status == 200
    assert response.body == b"done"


@pytest.mark.asyncio
async def test_read_timeout_raises_client_error() -> None:
    accepted: list[asyncio.StreamWriter] = []

    def _accept_silently(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        accepted.append(writer)

    silent_server = await asyncio.start_server(
        _accept_silently, "127.0.0.1", 0
    )
    port = silent_server.sockets[0].getsockname()[1]
    client = AsyncioClient(read_timeout=0.05)

    try:
        with pytest.raises(HTTPClientError) as exc_info:
            await client.send(
                HTTPRequest(method="GET", target=f"http://127.0.0.1:{port}/")
            )
    finally:
        for writer in accepted:
            writer.close()
        silent_server.close()
        await silent_server.wait_closed()

    assert f"exchange with 127.0.0.1:{port} failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_connection_refused_raises_client_error() -> None:
    port = _unused_port()
    client = AsyncioClient()

    with pytest.raises(HTTPClientError) as exc_info:
        await client.send(
            HTTPRequest(method="GET", target=f"http://127.0.0.1:{port}/")
        )

    assert f"failed to connect to 127.0.0.1:{port}" in str(exc_info.value)


@pytest.mark.asyncio
async def test_non_absolute_target_raises_client_error() -> None:
    client = AsyncioClient()

    with pytest.raises(HTTPClientError) as exc_info:
        await client.send(HTTPRequest(method="GET", target="/relative"))

    assert "not absolute-form" in str(exc_info.value)


@pytest.mark.asyncio
async def test_unparseable_response_raises_client_error() -> None:
    async with OneShotServer(b"not http at all\r\n\r\n") as upstream:
        client = AsyncioClient()

        with pytest.raises(HTTPClientError) as exc_info:
            await client.send(
                HTTPRequest(
                    method="GET",
                    target=f"http://127.0.0.1:{upstream.port}/",
                )
            )

    assert "failed to parse upstream response" in str(exc_info.value)
