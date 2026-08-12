from __future__ import annotations

import asyncio
import gzip
import socket

import httpx
import pytest
import pytest_asyncio

from localstub.cli import parse_args, run_http_proxy
from localstub.forward import RawForwarder
from localstub.http.clients.asyncio import AsyncioClient
from localstub.middleware import ResponderContext, ResponderNext, ResponseSpec
from localstub.server import AsyncHTTPTestServer, HTTPResponse


def _parse_response_headers(header_bytes: bytes) -> dict[bytes, bytes]:
    headers: dict[bytes, bytes] = {}
    for line in header_bytes.split(b"\r\n")[1:]:
        if not line:
            break
        name, separator, value = line.partition(b":")
        if not separator:
            continue
        headers[name.strip().lower()] = value.strip()
    return headers


def _is_transfer_encoding_chunked(transfer_encoding: bytes) -> bool:
    return any(
        token.strip().lower() == b"chunked"
        for token in transfer_encoding.split(b",")
    )


async def _read_chunked_body_bytes(reader: asyncio.StreamReader) -> bytes:
    body_bytes = bytearray()

    while True:
        size_line = await reader.readuntil(b"\r\n")
        body_bytes.extend(size_line)

        size_value = size_line[:-2].split(b";", 1)[0].strip()
        chunk_size = int(size_value, 16) if size_value else 0

        if chunk_size == 0:
            while True:
                trailer_line = await reader.readuntil(b"\r\n")
                body_bytes.extend(trailer_line)
                if trailer_line == b"\r\n":
                    return bytes(body_bytes)

        data_with_crlf = await reader.readexactly(chunk_size + 2)
        body_bytes.extend(data_with_crlf)


async def _read_http_response_bytes(
    reader: asyncio.StreamReader,
    *,
    timeout: float = 1.0,
) -> bytes:
    async with asyncio.timeout(timeout):
        header_bytes = await reader.readuntil(b"\r\n\r\n")
        headers = _parse_response_headers(header_bytes)

        transfer_encoding = headers.get(b"transfer-encoding", b"")
        if _is_transfer_encoding_chunked(transfer_encoding):
            return header_bytes + await _read_chunked_body_bytes(reader)

        content_length = headers.get(b"content-length")
        if content_length is None:
            return header_bytes + await reader.read()

        return header_bytes + await reader.readexactly(int(content_length))


async def _connect_when_ready(
    host: str,
    port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    async with asyncio.timeout(2.0):
        while True:
            try:
                return await asyncio.open_connection(host, port)
            except OSError:
                await asyncio.sleep(0.01)


@pytest_asyncio.fixture
async def upstream_server() -> AsyncHTTPTestServer:
    """Create an upstream server that the proxy will forward to."""
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"upstream": True, "status": "ok"})
        yield server


@pytest.mark.asyncio
async def test_http_proxy_mode_handles_expect_100_continue_client(
    upstream_server: AsyncHTTPTestServer,
) -> None:
    with socket.socket() as available_port:
        available_port.bind(("127.0.0.1", 0))
        proxy_port = available_port.getsockname()[1]

    args = parse_args([
        "--mode",
        "http-proxy",
        "--port",
        str(proxy_port),
    ])
    proxy_task = asyncio.create_task(run_http_proxy(args))

    try:
        reader, writer = await _connect_when_ready("127.0.0.1", proxy_port)
        try:
            body = b"expect continue upload"
            request_headers = (
                f"PUT {upstream_server.url}upload HTTP/1.1\r\n"
                f"Host: {upstream_server.host}:{upstream_server.port}\r\n"
                f"Content-Length: {len(body)}\r\n"
                "Expect: 100-continue\r\n"
                "Connection: close\r\n"
                "\r\n"
            ).encode()
            writer.write(request_headers)
            await writer.drain()

            interim_response = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                timeout=0.5,
            )
            assert interim_response == b"HTTP/1.1 100 Continue\r\n\r\n"

            writer.write(body)
            await writer.drain()
            final_response = await _read_http_response_bytes(reader)

            assert b"HTTP/1.1 200 OK" in final_response
            upstream_request = await upstream_server.next_request(timeout=0.5)
            assert upstream_request.body == body
        finally:
            writer.close()
            await writer.wait_closed()
    finally:
        proxy_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await proxy_task


@pytest.mark.asyncio
async def test_http_proxy_mode_ignores_environment_proxy(
    upstream_server: AsyncHTTPTestServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with AsyncHTTPTestServer() as environment_proxy:
        environment_proxy.set_text_response(
            "request used environment proxy",
            status=418,
        )
        monkeypatch.setenv("HTTP_PROXY", environment_proxy.url)
        monkeypatch.setenv("http_proxy", environment_proxy.url)
        monkeypatch.setenv("NO_PROXY", "")
        monkeypatch.setenv("no_proxy", "")

        with socket.socket() as available_port:
            available_port.bind(("127.0.0.1", 0))
            proxy_port = available_port.getsockname()[1]

        args = parse_args([
            "--mode",
            "http-proxy",
            "--port",
            str(proxy_port),
        ])
        proxy_task = asyncio.create_task(run_http_proxy(args))

        try:
            reader, writer = await _connect_when_ready(
                "127.0.0.1",
                proxy_port,
            )
            try:
                request = (
                    f"GET {upstream_server.url} HTTP/1.1\r\n"
                    f"Host: {upstream_server.host}:"
                    f"{upstream_server.port}\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                ).encode()
                writer.write(request)
                await writer.drain()

                response = await _read_http_response_bytes(reader)

                assert b"HTTP/1.1 200 OK" in response
                assert b'"upstream": true' in response
                await upstream_server.next_request(timeout=0.5)
            finally:
                writer.close()
                await writer.wait_closed()
        finally:
            proxy_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await proxy_task


class TestProxyRequestRecording:
    """Tests for recording proxy requests with absolute-form URIs."""

    @pytest.mark.asyncio
    async def test_server_records_absolute_uri_in_path(self) -> None:
        async with AsyncHTTPTestServer() as server:
            server.set_json_response({"mocked": True})

            # Send a request with absolute-form URI
            reader, writer = await asyncio.open_connection(
                server.host, server.port
            )
            try:
                request = (
                    b"GET http://example.com/api/users HTTP/1.1\r\n"
                    b"Host: example.com\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                )
                writer.write(request)
                await writer.drain()

                # Read response
                response = await _read_http_response_bytes(reader)
                assert b"200 OK" in response
            finally:
                writer.close()
                await writer.wait_closed()

            # Verify the request was recorded with full URI
            assert server.last_request is not None
            assert server.last_request.target == "http://example.com/api/users"
            assert server.last_request.is_proxy_request

    @pytest.mark.asyncio
    async def test_effective_path_for_routing(self) -> None:
        async with AsyncHTTPTestServer() as server:
            # Add a route for /api/users
            async def users_handler(request):
                return HTTPResponse.json({"handler": "users"})

            server.add_route("GET", "/api/users", users_handler)

            # Send request with absolute-form URI
            reader, writer = await asyncio.open_connection(
                server.host, server.port
            )
            try:
                request = (
                    b"GET http://example.com/api/users HTTP/1.1\r\n"
                    b"Host: example.com\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                )
                writer.write(request)
                await writer.drain()

                response = await _read_http_response_bytes(reader)
                assert b"200 OK" in response
                assert b'"handler": "users"' in response
            finally:
                writer.close()
                await writer.wait_closed()


class TestProxyForwarding:
    """Tests for forwarding proxy requests to upstream."""

    @pytest.mark.asyncio
    async def test_forwarder_does_not_corrupt_compressed_response(
        self,
    ) -> None:
        body = b"compressed upstream response"
        compressed_body = gzip.compress(body)

        async with AsyncHTTPTestServer() as upstream:
            upstream.set_raw_response(
                compressed_body,
                headers={"Content-Encoding": "gzip"},
            )

            async with (
                AsyncHTTPTestServer(upstream_client=AsyncioClient()) as proxy,
                httpx.AsyncClient(proxy=proxy.url) as downstream_client,
            ):
                response = await downstream_client.get(
                    upstream.url,
                    headers={"Connection": "close"},
                )

        assert response.content == body

    @pytest.mark.asyncio
    async def test_forwards_to_upstream_with_forwarder(
        self, upstream_server: AsyncHTTPTestServer
    ) -> None:
        async with AsyncHTTPTestServer(
            upstream_client=AsyncioClient()
        ) as proxy:
            # Send request through the proxy
            reader, writer = await asyncio.open_connection(
                proxy.host, proxy.port
            )
            try:
                # Request to the upstream server URL
                request = (
                    f"GET {upstream_server.url} HTTP/1.1\r\n"
                    f"Host: {upstream_server.host}\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                ).encode()
                writer.write(request)
                await writer.drain()

                response = await _read_http_response_bytes(reader)
                assert b"200 OK" in response
                assert b'"upstream": true' in response
            finally:
                writer.close()
                await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_forwarder_preserves_non_utf8_request_body(self) -> None:
        captured_body = b""
        request_received = asyncio.Event()

        async def upstream_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            nonlocal captured_body
            headers = await reader.readuntil(b"\r\n\r\n")
            content_length = 0
            for line in headers.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    content_length = int(line.split(b":", 1)[1].strip())
                    break

            if content_length:
                captured_body = await reader.read(content_length)
            else:
                captured_body = b""

            request_received.set()
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 0\r\n"
                b"Connection: close\r\n"
                b"\r\n"
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
        assert upstream.sockets is not None
        addr = upstream.sockets[0].getsockname()
        upstream_host, upstream_port = addr[0], addr[1]

        try:
            body = b"\xff\xfe\xfd\x00abc"
            async with AsyncHTTPTestServer(
                upstream_client=AsyncioClient()
            ) as proxy:
                reader, writer = await asyncio.open_connection(
                    proxy.host, proxy.port
                )
                try:
                    request = (
                        f"POST http://{upstream_host}:{upstream_port}"
                        "/upload HTTP/1.1\r\n"
                        f"Host: {upstream_host}:{upstream_port}\r\n"
                        f"Content-Length: {len(body)}\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                    ).encode() + body
                    writer.write(request)
                    await writer.drain()

                    await _read_http_response_bytes(reader)
                    await asyncio.wait_for(
                        request_received.wait(), timeout=1.0
                    )
                finally:
                    writer.close()
                    await writer.wait_closed()
        finally:
            upstream.close()
            await upstream.wait_closed()

        assert captured_body == body

    @pytest.mark.asyncio
    async def test_record_mode_without_forwarder(self) -> None:
        async with AsyncHTTPTestServer() as server:
            server.set_json_response({"mocked": True})

            reader, writer = await asyncio.open_connection(
                server.host, server.port
            )
            try:
                request = (
                    b"GET http://example.com/api HTTP/1.1\r\n"
                    b"Host: example.com\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                )
                writer.write(request)
                await writer.drain()

                response = await _read_http_response_bytes(reader)
                assert b"200 OK" in response
                assert b'"mocked": true' in response
            finally:
                writer.close()
                await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_response_sequence_overrides_forwarding(
        self, upstream_server: AsyncHTTPTestServer
    ) -> None:
        async with AsyncHTTPTestServer(
            upstream_client=AsyncioClient()
        ) as proxy:
            # Set a response sequence
            proxy.set_response_sequence([
                HTTPResponse.json({"sequence": 1}),
                HTTPResponse.json({"sequence": 2}),
            ])

            for i in range(1, 3):
                reader, writer = await asyncio.open_connection(
                    proxy.host, proxy.port
                )
                try:
                    request = (
                        f"GET {upstream_server.url} HTTP/1.1\r\n"
                        f"Host: {upstream_server.host}\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                    ).encode()
                    writer.write(request)
                    await writer.drain()

                    response = await _read_http_response_bytes(reader)
                    # Should get sequence response, not upstream
                    assert f'"sequence": {i}'.encode() in response
                finally:
                    writer.close()
                    await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_origin_form_not_forwarded(
        self, upstream_server: AsyncHTTPTestServer
    ) -> None:
        async with AsyncHTTPTestServer(
            upstream_client=AsyncioClient()
        ) as proxy:
            proxy.set_json_response({"local": True})

            reader, writer = await asyncio.open_connection(
                proxy.host, proxy.port
            )
            try:
                # Origin-form request (not absolute URI)
                request = (
                    b"GET /local/path HTTP/1.1\r\n"
                    b"Host: localhost\r\n"
                    b"Connection: close\r\n"
                    b"\r\n"
                )
                writer.write(request)
                await writer.drain()

                response = await _read_http_response_bytes(reader)
                # Should get local response, not forwarded
                assert b'"local": true' in response
            finally:
                writer.close()
                await writer.wait_closed()


class TestRawForwarding:
    """Tests for raw socket forwarding that preserves Transfer-Encoding."""

    @pytest.mark.asyncio
    async def test_preserves_chunked_request_framing(self) -> None:
        captured_headers = b""
        captured_body = b""
        request_received = asyncio.Event()

        async def upstream_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            nonlocal captured_headers, captured_body
            captured_headers = await reader.readuntil(b"\r\n\r\n")
            captured_body = await reader.read(1024)
            request_received.set()

            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
        addr = upstream.sockets[0].getsockname()
        upstream_host, upstream_port = addr[0], addr[1]

        try:
            forwarder = RawForwarder(verify_upstream=False)
            async with AsyncHTTPTestServer(raw_forwarder=forwarder) as proxy:
                reader, writer = await asyncio.open_connection(
                    proxy.host, proxy.port
                )
                try:
                    request = (
                        f"POST http://{upstream_host}:{upstream_port}"
                        "/upload HTTP/1.1\r\n"
                        f"Host: {upstream_host}:{upstream_port}\r\n"
                        "Transfer-Encoding: chunked\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                    ).encode() + b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
                    writer.write(request)
                    await writer.drain()

                    await _read_http_response_bytes(reader)

                    await asyncio.wait_for(
                        request_received.wait(), timeout=1.0
                    )
                finally:
                    writer.close()
                    await writer.wait_closed()
        finally:
            upstream.close()
            await upstream.wait_closed()

        assert b"Transfer-Encoding: chunked" in captured_headers
        assert captured_body.startswith(b"5\r\n")

    @pytest.mark.asyncio
    async def test_rejects_body_rewrites(self) -> None:
        request_received = asyncio.Event()

        async def upstream_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            request_received.set()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
        addr = upstream.sockets[0].getsockname()
        upstream_host, upstream_port = addr[0], addr[1]

        try:
            forwarder = RawForwarder(verify_upstream=False)
            async with AsyncHTTPTestServer(raw_forwarder=forwarder) as proxy:

                async def rewrite_body(
                    ctx: ResponderContext,
                    call_next: ResponderNext,
                ) -> ResponseSpec:
                    ctx2 = ctx.clone_request(body=b"HELLO")
                    return await call_next(ctx=ctx2)

                proxy.responder_middlewares.append(rewrite_body)

                reader, writer = await asyncio.open_connection(
                    proxy.host, proxy.port
                )
                try:
                    request = (
                        f"POST http://{upstream_host}:{upstream_port}"
                        "/upload HTTP/1.1\r\n"
                        f"Host: {upstream_host}:{upstream_port}\r\n"
                        "Content-Length: 5\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                        "hello"
                    ).encode()
                    writer.write(request)
                    await writer.drain()

                    response = await _read_http_response_bytes(reader)
                    assert b"500 Internal Server Error" in response
                    assert (
                        b"Raw proxy forwarding does not support request "
                        b"body rewrites."
                    ) in response
                finally:
                    writer.close()
                    await writer.wait_closed()
        finally:
            upstream.close()
            await upstream.wait_closed()

        assert not request_received.is_set()

    @pytest.mark.asyncio
    async def test_rejects_content_length_rewrites(self) -> None:
        request_received = asyncio.Event()

        async def upstream_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            headers = await reader.readuntil(b"\r\n\r\n")
            assert b"Content-Length: 6\r\n" in headers
            request_received.set()
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
        assert upstream.sockets is not None
        upstream_host, upstream_port = upstream.sockets[0].getsockname()[:2]

        try:
            forwarder = RawForwarder(verify_upstream=False)
            async with AsyncHTTPTestServer(raw_forwarder=forwarder) as proxy:

                async def rewrite_content_length(
                    ctx: ResponderContext,
                    call_next: ResponderNext,
                ) -> ResponseSpec:
                    next_ctx = ctx.clone_request(
                        headers={"Content-Length": "6"}
                    )
                    return await call_next(ctx=next_ctx)

                proxy.use(rewrite_content_length)

                reader, writer = await asyncio.open_connection(
                    proxy.host, proxy.port
                )
                try:
                    request = (
                        f"POST http://{upstream_host}:{upstream_port}"
                        "/upload HTTP/1.1\r\n"
                        f"Host: {upstream_host}:{upstream_port}\r\n"
                        "Content-Length: 5\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                        "hello"
                    ).encode()
                    writer.write(request)
                    await writer.drain()

                    response = await _read_http_response_bytes(reader)
                    assert b"500 Internal Server Error" in response
                finally:
                    writer.close()
                    await writer.wait_closed()
        finally:
            upstream.close()
            await upstream.wait_closed()

        assert not request_received.is_set()

    @pytest.mark.asyncio
    async def test_preserves_chunked_transfer_encoding(self) -> None:

        async def chunked_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            # Read and discard request
            await reader.readuntil(b"\r\n\r\n")
            # Send chunked response
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"Content-Type: text/plain\r\n"
                b"\r\n"
                b"5\r\nhello\r\n"
                b"6\r\n world\r\n"
                b"0\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        # Start raw upstream server
        upstream = await asyncio.start_server(chunked_handler, "127.0.0.1", 0)
        addr = upstream.sockets[0].getsockname()
        upstream_host, upstream_port = addr[0], addr[1]

        try:
            forwarder = RawForwarder(verify_upstream=False)
            async with AsyncHTTPTestServer(raw_forwarder=forwarder) as proxy:
                reader, writer = await asyncio.open_connection(
                    proxy.host, proxy.port
                )
                try:
                    # Send absolute-form proxy request
                    request = (
                        f"GET http://{upstream_host}:{upstream_port}/"
                        f" HTTP/1.1\r\n"
                        f"Host: {upstream_host}:{upstream_port}\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                    ).encode()
                    writer.write(request)
                    await writer.drain()

                    response = await _read_http_response_bytes(reader)

                    # Verify Transfer-Encoding is preserved
                    assert b"Transfer-Encoding: chunked" in response
                    # Verify chunked framing is preserved
                    assert b"5\r\nhello\r\n" in response
                    assert b"6\r\n world\r\n" in response
                    assert b"0\r\n\r\n" in response
                    # Verify Content-Length was NOT added
                    assert b"Content-Length" not in response
                finally:
                    writer.close()
                    await writer.wait_closed()
        finally:
            upstream.close()
            await upstream.wait_closed()

    @pytest.mark.asyncio
    async def test_preserves_content_length(self) -> None:

        async def content_length_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            await reader.readuntil(b"\r\n\r\n")
            body = b"hello world"
            response_bytes = (
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/plain\r\n"
                + f"Content-Length: {len(body)}\r\n".encode()
                + b"\r\n"
                + body
            )
            writer.write(response_bytes)
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(
            content_length_handler, "127.0.0.1", 0
        )
        addr = upstream.sockets[0].getsockname()
        upstream_host, upstream_port = addr[0], addr[1]

        try:
            forwarder = RawForwarder(verify_upstream=False)
            async with AsyncHTTPTestServer(raw_forwarder=forwarder) as proxy:
                reader, writer = await asyncio.open_connection(
                    proxy.host, proxy.port
                )
                try:
                    request = (
                        f"GET http://{upstream_host}:{upstream_port}/"
                        f" HTTP/1.1\r\n"
                        f"Host: {upstream_host}:{upstream_port}\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                    ).encode()
                    writer.write(request)
                    await writer.drain()

                    response = await _read_http_response_bytes(reader)

                    # Verify Content-Length is preserved
                    assert b"Content-Length: 11" in response
                    # Verify Transfer-Encoding was NOT added
                    assert b"Transfer-Encoding" not in response
                    assert b"hello world" in response
                finally:
                    writer.close()
                    await writer.wait_closed()
        finally:
            upstream.close()
            await upstream.wait_closed()

    @pytest.mark.asyncio
    async def test_closes_downstream_after_eof_delimited_response(
        self,
    ) -> None:
        body = b"eof-delimited response"

        async def eof_delimited_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\n" + body
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(
            eof_delimited_handler,
            "127.0.0.1",
            0,
        )
        assert upstream.sockets is not None
        upstream_host, upstream_port = upstream.sockets[0].getsockname()[:2]

        try:
            forwarder = RawForwarder(verify_upstream=False)
            async with AsyncHTTPTestServer(raw_forwarder=forwarder) as proxy:
                reader, writer = await asyncio.open_connection(
                    proxy.host,
                    proxy.port,
                )
                try:
                    request = (
                        f"GET http://{upstream_host}:{upstream_port}/"
                        " HTTP/1.1\r\n"
                        f"Host: {upstream_host}:{upstream_port}\r\n"
                        "\r\n"
                    ).encode()
                    writer.write(request)
                    await writer.drain()

                    response = await _read_http_response_bytes(
                        reader,
                        timeout=0.5,
                    )
                finally:
                    writer.close()
                    await writer.wait_closed()
        finally:
            upstream.close()
            await upstream.wait_closed()

        assert body in response

    @pytest.mark.asyncio
    async def test_recorded_response_has_wire_bytes(self) -> None:

        async def chunked_handler(
            reader: asyncio.StreamReader,
            writer: asyncio.StreamWriter,
        ) -> None:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Transfer-Encoding: chunked\r\n"
                b"\r\n"
                b"3\r\nfoo\r\n"
                b"0\r\n\r\n"
            )
            await writer.drain()
            writer.close()
            await writer.wait_closed()

        upstream = await asyncio.start_server(chunked_handler, "127.0.0.1", 0)
        addr = upstream.sockets[0].getsockname()
        upstream_host, upstream_port = addr[0], addr[1]

        try:
            forwarder = RawForwarder(verify_upstream=False)
            async with AsyncHTTPTestServer(raw_forwarder=forwarder) as proxy:
                reader, writer = await asyncio.open_connection(
                    proxy.host, proxy.port
                )
                try:
                    request = (
                        f"GET http://{upstream_host}:{upstream_port}/"
                        f" HTTP/1.1\r\n"
                        f"Host: {upstream_host}:{upstream_port}\r\n"
                        "Connection: close\r\n"
                        "\r\n"
                    ).encode()
                    writer.write(request)
                    await writer.drain()

                    # Read response to complete the request
                    await _read_http_response_bytes(reader)
                finally:
                    writer.close()
                    await writer.wait_closed()

                # Check recorded response
                recorded = await proxy.next_response(timeout=1.0)
                assert recorded.status == 200
                assert b"Transfer-Encoding: chunked" in recorded.wire_raw_bytes
                assert b"3\r\nfoo\r\n" in recorded.wire_raw_bytes
        finally:
            upstream.close()
            await upstream.wait_closed()
