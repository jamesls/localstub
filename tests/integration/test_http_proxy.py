"""Integration tests for HTTP forward proxy functionality."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

from localstub.forward import Forwarder
from localstub.server import AsyncHTTPTestServer, HTTPResponse


@pytest_asyncio.fixture
async def upstream_server() -> AsyncHTTPTestServer:
    """Create an upstream server that the proxy will forward to."""
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"upstream": True, "status": "ok"})
        yield server


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
                response = await reader.read(4096)
                assert b"200 OK" in response
            finally:
                writer.close()
                await writer.wait_closed()

            # Verify the request was recorded with full URI
            assert server.last_request is not None
            assert server.last_request.path == "http://example.com/api/users"
            assert server.last_request.is_proxy_request is True

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

                response = await reader.read(4096)
                assert b"200 OK" in response
                assert b'"handler": "users"' in response
            finally:
                writer.close()
                await writer.wait_closed()


class TestProxyForwarding:
    """Tests for forwarding proxy requests to upstream."""

    @pytest.mark.asyncio
    async def test_forwards_to_upstream_with_forwarder(
        self, upstream_server: AsyncHTTPTestServer
    ) -> None:
        async with httpx.AsyncClient() as http_client:
            async with AsyncHTTPTestServer(
                proxy_forwarder=http_client
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

                    response = await reader.read(4096)
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
            async with httpx.AsyncClient() as http_client:
                async with AsyncHTTPTestServer(
                    proxy_forwarder=http_client
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

                        await reader.read(4096)
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

                response = await reader.read(4096)
                assert b"200 OK" in response
                assert b'"mocked": true' in response
            finally:
                writer.close()
                await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_response_sequence_overrides_forwarding(
        self, upstream_server: AsyncHTTPTestServer
    ) -> None:
        async with httpx.AsyncClient() as http_client:
            async with AsyncHTTPTestServer(
                proxy_forwarder=http_client
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

                        response = await reader.read(4096)
                        # Should get sequence response, not upstream
                        assert f'"sequence": {i}'.encode() in response
                    finally:
                        writer.close()
                        await writer.wait_closed()

    @pytest.mark.asyncio
    async def test_origin_form_not_forwarded(
        self, upstream_server: AsyncHTTPTestServer
    ) -> None:
        async with httpx.AsyncClient() as http_client:
            async with AsyncHTTPTestServer(
                proxy_forwarder=http_client
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

                    response = await reader.read(4096)
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
            forwarder = Forwarder(verify_upstream=False)
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

                    await reader.read(4096)

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
            forwarder = Forwarder(verify_upstream=False)
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

                    response = await reader.read(4096)

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
            forwarder = Forwarder(verify_upstream=False)
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

                    response = await reader.read(4096)

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
            forwarder = Forwarder(verify_upstream=False)
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
                    await reader.read(4096)
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
