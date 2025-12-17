import httpx
import asyncio
import gzip
import ssl

import trustme
import pytest

from localstub.server import AsyncHTTPTestServer
from localstub.tlsproxy import AsyncTLSInterceptProxy


@pytest.mark.asyncio
async def test_tls_proxy_intercepts_https_request_to_localstub():
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"ok": True})

        async with AsyncTLSInterceptProxy(server=server) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_path = proxy.ca.ca_pem_path()
            verify_ctx = ssl.create_default_context(cafile=str(verify_path))

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get("https://example.com/")

            assert response.status_code == 200
            assert response.json() == {"ok": True}

        # The intercepted request should be recorded by the underlying server
        request = await server.next_request(timeout=1.0)
        assert request.path == "/"
        assert request.headers is not None
        assert request.headers["host"] == "example.com"


@pytest.mark.asyncio
async def test_tls_proxy_parses_ipv6_connect_target():
    async with AsyncHTTPTestServer() as server:
        server.set_text_response("ipv6")

        async with AsyncTLSInterceptProxy(server=server) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    "https://[2001:db8::1]/ipv6",
                )

        assert response.status_code == 200
        assert response.text == "ipv6"

        request = await server.next_request(timeout=1.0)
        assert request.path == "/ipv6"
        assert request.headers is not None
        assert "2001:db8::1" in request.headers["host"]


def test_proxy_address_raises_before_start():
    proxy = AsyncTLSInterceptProxy()
    with pytest.raises(RuntimeError):
        _ = proxy.address


@pytest.mark.asyncio
async def test_next_request_and_response_timeout_raise():
    proxy = AsyncTLSInterceptProxy()

    with pytest.raises(asyncio.TimeoutError):
        await proxy.next_request(timeout=0.01)

    with pytest.raises(asyncio.TimeoutError):
        await proxy.next_response(timeout=0.01)


@pytest.mark.asyncio
async def test_start_idempotent_and_aclose_noop():
    proxy = AsyncTLSInterceptProxy()

    async with proxy:
        first_host, first_port = proxy.address
        await proxy.start()
        assert proxy.address == (first_host, first_port)

    await proxy.aclose()


@pytest.mark.asyncio
async def test_non_connect_request_returns_400():
    async with AsyncTLSInterceptProxy() as proxy:
        host, port = proxy.address
        reader, writer = await asyncio.open_connection(host, port)

        writer.write(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
        await writer.drain()

        response = await reader.read(1024)
        assert b"400 Bad Request" in response

        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_intercept_without_server_returns_502():
    async with AsyncTLSInterceptProxy(server=None) as proxy:
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            response = await client.get("https://example.com/none")

    assert response.status_code == 502


async def _close_immediately(
    _reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    writer.write(b"\r\n\r\n")
    await writer.drain()
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_forward_returns_502_when_upstream_closes_early():
    server = await asyncio.start_server(
        _close_immediately,
        "127.0.0.1",
        0,
    )
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{upstream_host}:{upstream_port}/early",
                    follow_redirects=False,
                )

        assert response.status_code == 502

        with pytest.raises(asyncio.TimeoutError):
            await proxy.next_response(timeout=0.1)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_returns_502_when_upstream_connection_fails():
    # Acquire an unused local port, then close the server so that connecting
    # to the port will be refused.
    tmp = await asyncio.start_server(lambda r, w: None, "127.0.0.1", 0)
    unused_port = tmp.sockets[0].getsockname()[1]
    tmp.close()
    await tmp.wait_closed()

    async with AsyncTLSInterceptProxy(
        server=None,
        default_mode="forward",
        verify_upstream=False,
        upstream_tls=False,
    ) as proxy:
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            response = await client.get(
                f"https://127.0.0.1:{unused_port}/unreachable",
                follow_redirects=False,
            )

    assert response.status_code == 502

    # The client request should still be recorded even though no upstream
    # connection could be established.
    recorded_request = await proxy.next_request(timeout=1.0)
    assert recorded_request.path == "/unreachable"
    assert recorded_request.headers is not None
    assert "127.0.0.1" in recorded_request.headers["host"]

    # No upstream response should be recorded.
    with pytest.raises(asyncio.TimeoutError):
        await proxy.next_response(timeout=0.1)


async def _chunked_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    await reader.readuntil(b"\r\n\r\n")
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"Trailer: X-Trail\r\n"
        b"Content-Type: text/plain\r\n"
        b"\r\n"
        b"4\r\npeek\r\n"
        b"3\r\nboo\r\n"
        b"0\r\nX-Trail: done\r\n\r\n"
    )
    await writer.drain()
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_forward_records_chunked_response_with_trailer():
    server = await asyncio.start_server(
        _chunked_handler,
        "127.0.0.1",
        0,
    )
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{upstream_host}:{upstream_port}/chunked"
                )

        assert response.status_code == 200
        assert response.text == "peekboo"

        recorded = await proxy.next_response(timeout=1.0)
        assert recorded.body == "peekboo"
        assert b"X-Trail: done" in recorded.wire_raw_bytes
    finally:
        server.close()
        await server.wait_closed()


def _gzip_handler(body: bytes, invalid: bool = False):
    async def handler(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        payload = body if invalid else gzip.compress(body)
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            + f"Content-Length: {len(payload)}\r\n".encode()
            + b"Content-Encoding: gzip\r\n"
            + b"Content-Type: text/plain\r\n\r\n"
            + payload
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    return handler


@pytest.mark.asyncio
async def test_forward_decompresses_gzip_response():
    body = b"hello gzip"
    server = await asyncio.start_server(
        _gzip_handler(body),
        "127.0.0.1",
        0,
    )
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{upstream_host}:{upstream_port}/gzip"
                )

        assert response.status_code == 200
        assert response.text == "hello gzip"

        recorded = await proxy.next_response(timeout=1.0)
        assert recorded.body == "hello gzip"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_returns_raw_body_when_gzip_invalid():
    body = b"not gzipped"
    server = await asyncio.start_server(
        _gzip_handler(body, invalid=True),
        "127.0.0.1",
        0,
    )
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                with pytest.raises(httpx.DecodingError):
                    await client.get(
                        f"https://{upstream_host}:{upstream_port}/invalid-gzip"
                    )

        recorded = await proxy.next_response(timeout=1.0)
        assert recorded.body == "not gzipped"
    finally:
        server.close()
        await server.wait_closed()


async def _content_length_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    request_line = await reader.readline()
    if not request_line:
        writer.close()
        await writer.wait_closed()
        return

    await reader.readuntil(b"\r\n\r\n")
    path = request_line.decode("ascii", errors="replace").split(" ")[1]

    if path == "/zero":
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 0\r\n"
            b"Content-Type: text/plain\r\n\r\n"
        )
    else:
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: nope\r\n"
            b"Content-Type: text/plain\r\n\r\n"
        )
    await writer.drain()
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_forward_handles_zero_and_invalid_content_length():
    server = await asyncio.start_server(
        _content_length_handler,
        "127.0.0.1",
        0,
    )
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as client:
                zero_response = await client.get(
                    f"https://{upstream_host}:{upstream_port}/zero"
                )
                invalid_response = await client.get(
                    f"https://{upstream_host}:{upstream_port}/invalid"
                )

        assert zero_response.status_code == 200
        assert zero_response.text == ""
        assert invalid_response.status_code == 502

        first_recorded = await proxy.next_response(timeout=1.0)
        assert first_recorded.body == ""

        with pytest.raises(asyncio.TimeoutError):
            await proxy.next_response(timeout=0.1)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_to_plain_http_when_tls_disabled():
    """Forward mode: proxy relays to HTTP upstream when TLS is disabled."""

    async with AsyncHTTPTestServer() as upstream:
        upstream.set_json_response({"upstream": True})

        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            upstream_host, upstream_port = upstream.host, upstream.port
            assert upstream_host is not None
            assert upstream_port is not None

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{upstream_host}:{upstream_port}/forward",
                    follow_redirects=False,
                )

        recorded = await upstream.next_request(timeout=1.0)
        assert recorded.path == "/forward"
        assert response.status_code == 200
        assert response.json() == {"upstream": True}


@pytest.mark.asyncio
async def test_forward_uses_tls_when_verify_disabled_custom_port():
    ca = trustme.CA()
    cert = ca.issue_cert("localhost", "127.0.0.1")
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert.configure_cert(server_ctx)
    server_ctx.set_alpn_protocols(["http/1.1"])
    server_ctx.options |= ssl.OP_NO_COMPRESSION

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        body = b"tls-ok"
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 6\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n" + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(
        handle,
        "127.0.0.1",
        0,
        ssl=server_ctx,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]
    assert server_port != 443

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=True,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/tls"
                )

        assert response.status_code == 200
        assert response.text == "tls-ok"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        assert recorded_response.body == "tls-ok"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_preserves_close_delimited_response_bodies():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        body = b"streamed close body"
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n" + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/close-body"
                )

        assert response.status_code == 200
        assert response.text == "streamed close body"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.body == "streamed close body"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_proxy_records_request_and_response_when_forwarding_to_real():
    """
    Proxy should record the decrypted HTTP request and the upstream HTTP
    response while forwarding to a real HTTPS origin.
    """

    async with AsyncTLSInterceptProxy(default_mode="forward") as proxy:
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            response = await client.get("https://example.com/")

    assert response.status_code == 200
    assert "Example Domain" in response.text

    recorded_request = await proxy.next_request(timeout=2.0)
    assert recorded_request.path == "/"
    assert recorded_request.headers is not None
    assert recorded_request.headers["host"] == "example.com"

    recorded_response = await proxy.next_response(timeout=2.0)
    assert recorded_response.status == 200
    assert recorded_response.body is not None
    assert "Example Domain" in recorded_response.body


@pytest.mark.asyncio
async def test_forward_handles_head_request_with_content_length():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request_line = await reader.readline()
        await reader.readuntil(b"\r\n\r\n")

        method = request_line.decode("ascii", errors="replace").split(" ")[0]
        # For HEAD, respond with Content-Length but no body
        if method.upper() == "HEAD":
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 12345\r\n"
                b"Content-Type: text/html\r\n"
                b"\r\n"
            )
        else:
            body = b"Hello World!"
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 12\r\n"
                b"Content-Type: text/plain\r\n"
                b"\r\n" + body
            )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
                timeout=httpx.Timeout(5.0),
            ) as client:
                # HEAD request should complete quickly, not timeout
                response = await client.head(
                    f"https://{server_host}:{server_port}/authors.html"
                )

        assert response.status_code == 200
        assert response.headers.get("content-length") == "12345"
        # HEAD responses have no body
        assert response.content == b""

        recorded_request = await proxy.next_request(timeout=1.0)
        assert recorded_request.method == "HEAD"
        assert recorded_request.path == "/authors.html"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        # The recorded response body should be empty for HEAD
        assert recorded_response.body == ""
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_handles_client_closing_connection_early():
    """
    Proxy should gracefully handle client closing connection
    after reading the response.

    This simulates behavior seen with the AWS CLI where the client
    reads the response and closes the connection, which can cause
    ConnectionResetError in wait_closed().
    """

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        # Send a chunked response similar to what S3 returns
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"Content-Type: application/xml\r\n"
            b"\r\n"
            b"5\r\nhello\r\n"
            b"0\r\n\r\n"
        )
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            # Use httpx with no keepalive to ensure connection is closed
            # immediately after response is received
            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/test"
                )

        # Verify the proxy completed without error
        assert response.status_code == 200
        assert response.text == "hello"

        # The proxy should have recorded the request and response
        recorded_request = await proxy.next_request(timeout=1.0)
        assert recorded_request.path == "/test"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        assert recorded_response.body == "hello"
    finally:
        server.close()
        await server.wait_closed()


async def _expect_continue_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Handler that sends 100 Continue before the final response."""
    # Read full request including body
    await reader.readuntil(b"\r\n\r\n")

    # Send 100 Continue first
    writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
    await writer.drain()

    # Then send the final response
    body = b"upload accepted"
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 15\r\n"
        b"Content-Type: text/plain\r\n"
        b"\r\n" + body
    )
    await writer.drain()
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_forward_handles_100_continue_before_final_response():
    """
    Proxy should handle servers that send 100 Continue before the final
    response, relaying both to the client without closing prematurely.
    """
    server = await asyncio.start_server(
        _expect_continue_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.put(
                    f"https://{server_host}:{server_port}/upload",
                    content=b"test data",
                    headers={"Expect": "100-continue"},
                )

        assert response.status_code == 200
        assert response.text == "upload accepted"

        recorded_request = await proxy.next_request(timeout=1.0)
        assert recorded_request.method == "PUT"
        assert recorded_request.path == "/upload"

        # Only the final response should be recorded
        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        assert recorded_response.body == "upload accepted"
    finally:
        server.close()
        await server.wait_closed()


async def _multiple_informational_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Handler that sends multiple 1xx responses before final response."""
    await reader.readuntil(b"\r\n\r\n")

    # Send 100 Continue
    writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
    await writer.drain()

    # Send 102 Processing (used by WebDAV for long operations)
    writer.write(b"HTTP/1.1 102 Processing\r\n\r\n")
    await writer.drain()

    # Send final response
    body = b"done"
    writer.write(
        b"HTTP/1.1 201 Created\r\n"
        b"Content-Length: 4\r\n"
        b"Content-Type: text/plain\r\n"
        b"\r\n" + body
    )
    await writer.drain()
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_forward_handles_multiple_informational_responses():
    """
    Proxy should handle multiple 1xx informational responses before the
    final response, relaying each to the client.
    """
    server = await asyncio.start_server(
        _multiple_informational_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.post(
                    f"https://{server_host}:{server_port}/long-operation",
                    content=b"data",
                )

        assert response.status_code == 201
        assert response.text == "done"

        recorded_request = await proxy.next_request(timeout=1.0)
        assert recorded_request.method == "POST"
        assert recorded_request.path == "/long-operation"

        # Only the final response should be recorded
        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 201
        assert recorded_response.body == "done"
    finally:
        server.close()
        await server.wait_closed()


async def _chunked_with_continue_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Handler simulating S3's response to chunked uploads with Expect."""
    await reader.readuntil(b"\r\n\r\n")

    # Send 100 Continue
    writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
    await writer.drain()

    # Send chunked final response (like S3 does)
    # Chunk size b (hex) = 11 decimal = len("<Success/>\n")
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"Content-Type: application/xml\r\n"
        b"\r\n"
        b"b\r\n<Success/>\n\r\n"
        b"0\r\n\r\n"
    )
    await writer.drain()
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_forward_handles_100_continue_with_chunked_response():
    """
    Proxy should handle 100 Continue followed by a chunked response,
    which is the pattern used by S3 for uploads with Expect: 100-continue.
    """
    server = await asyncio.start_server(
        _chunked_with_continue_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.put(
                    f"https://{server_host}:{server_port}/bucket/key",
                    content=b"file contents",
                    headers={"Expect": "100-continue"},
                )

        assert response.status_code == 200
        assert response.text == "<Success/>\n"

        recorded_request = await proxy.next_request(timeout=1.0)
        assert recorded_request.method == "PUT"
        assert recorded_request.path == "/bucket/key"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        assert recorded_response.body == "<Success/>\n"
    finally:
        server.close()
        await server.wait_closed()


# ----------------------------------------------------------------------
# Response Transformation Tests
# ----------------------------------------------------------------------


async def _simple_json_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Handler that returns a simple JSON response."""
    await reader.readuntil(b"\r\n\r\n")
    body = b'{"message":"hello","count":42}'
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 30\r\n"
        b"Content-Type: application/json\r\n"
        b"\r\n" + body
    )
    await writer.drain()
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass


@pytest.mark.asyncio
async def test_forward_transforms_response_body_with_byteflip():
    """Transformer can flip bits in the response body using ByteFlip."""
    from localstub.server import ByteFlip
    from localstub.tlsproxy import fault_step_transformer

    server = await asyncio.start_server(
        _simple_json_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=fault_step_transformer(ByteFlip(offset=2)),
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/test"
                )

        # The response body should have byte at offset 2 flipped
        # Original: {"message":"hello","count":42}
        # Byte 2 is 'm' (0x6d), XOR with 0xFF = 0x92
        assert response.status_code == 200
        body = response.content
        assert body[2] != ord("m")  # The byte should be flipped

        # The recorded response should contain the original body
        recorded = await proxy.next_response(timeout=1.0)
        assert recorded.body == '{"message":"hello","count":42}'
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_transformer_delay_before():
    """Transformer can add delay before sending the response."""
    import time
    from localstub.tlsproxy import TransformResult, UpstreamResponse

    def delay_transformer(upstream: UpstreamResponse) -> TransformResult:
        return TransformResult(body=upstream.body, delay_before=0.1)

    server = await asyncio.start_server(
        _simple_json_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=delay_transformer,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            start = time.monotonic()
            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/test"
                )
            elapsed = time.monotonic() - start

        assert response.status_code == 200
        # Should have taken at least 100ms due to the delay
        assert elapsed >= 0.1
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_transformer_override_response():
    """Transformer can completely replace the response."""
    from localstub.server import HTTPResponse
    from localstub.tlsproxy import TransformResult, UpstreamResponse

    def override_transformer(upstream: UpstreamResponse) -> TransformResult:
        return TransformResult(
            override_response=HTTPResponse(
                status=503,
                headers={"Content-Type": "text/plain"},
                body="Service Unavailable",
            )
        )

    server = await asyncio.start_server(
        _simple_json_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=override_transformer,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/test"
                )

        # Should receive the overridden response
        assert response.status_code == 503
        assert response.text == "Service Unavailable"

        # The recorded response should contain the original upstream response
        recorded = await proxy.next_response(timeout=1.0)
        assert recorded.status == 200
        assert recorded.body == '{"message":"hello","count":42}'
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_async_transformer():
    """Transformer can be an async function."""
    from localstub.tlsproxy import TransformResult, UpstreamResponse

    async def async_transformer(upstream: UpstreamResponse) -> TransformResult:
        # Simulate some async work
        await asyncio.sleep(0.01)
        # Uppercase the body
        return TransformResult(body=upstream.body.upper())

    server = await asyncio.start_server(
        _simple_json_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=async_transformer,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/test"
                )

        # Body should be uppercased
        assert response.status_code == 200
        assert response.text == '{"MESSAGE":"HELLO","COUNT":42}'
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_fault_step_transformer_chains_multiple_steps():
    """fault_step_transformer can chain multiple FaultSteps."""
    from localstub.server import ByteFlip, TruncateBody
    from localstub.tlsproxy import fault_step_transformer

    server = await asyncio.start_server(
        _simple_json_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        # Chain: first truncate to 10 bytes, then flip byte at offset 0
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=fault_step_transformer(
                TruncateBody(keep_bytes=10),
                ByteFlip(offset=0),
            ),
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/test"
                )

        # Original: {"message":"hello","count":42}
        # Truncated to 10: {"message"
        # Then byte 0 '{' (0x7b) XOR 0xFF = 0x84
        body = response.content
        assert len(body) == 10
        assert body[0] != ord("{")  # First byte flipped
        assert body[1:] == b'"message"'
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_transformer_passthrough_when_none_returned():
    """Transformer returning None body passes through original response."""
    from localstub.tlsproxy import TransformResult, UpstreamResponse

    def passthrough_transformer(upstream: UpstreamResponse) -> TransformResult:
        # Return empty result - should passthrough original
        return TransformResult()

    server = await asyncio.start_server(
        _simple_json_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=passthrough_transformer,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/test"
                )

        # Should receive original response unchanged
        assert response.status_code == 200
        assert response.json() == {"message": "hello", "count": 42}
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_transformer_conditional_based_on_content_type():
    """Transformer can conditionally modify based on response headers."""
    from localstub.tlsproxy import TransformResult, UpstreamResponse

    def conditional_transformer(upstream: UpstreamResponse) -> TransformResult:
        content_type = ""
        if upstream.headers:
            content_type = upstream.headers.get("Content-Type", "")
        if "application/json" in content_type:
            # Corrupt JSON responses
            return TransformResult(body=b"corrupted!")
        # Pass through non-JSON responses
        return TransformResult(body=upstream.body)

    server = await asyncio.start_server(
        _simple_json_handler,
        "127.0.0.1",
        0,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=conditional_transformer,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/test"
                )

        # JSON response should be corrupted
        assert response.status_code == 200
        assert response.text == "corrupted!"
    finally:
        server.close()
        await server.wait_closed()
