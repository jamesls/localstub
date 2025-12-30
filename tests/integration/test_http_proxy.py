"""Integration tests for HTTP forward proxy functionality."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import pytest_asyncio

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
        """Server should record full absolute URI in request.path."""
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
        """Router should match routes using effective_path."""
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
        """When proxy_forwarder is set, forwards proxy requests upstream."""
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
    async def test_record_mode_without_forwarder(self) -> None:
        """Without forwarder, returns configured mock response."""
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
        """Response sequence takes priority over proxy forwarding."""
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
        """Origin-form requests are not forwarded even with forwarder."""
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
