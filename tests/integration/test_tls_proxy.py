import httpx
import asyncio
import gzip
import ssl

import trustme
import pytest

from localstub.server import AsyncHTTPTestServer
from localstub.tls_proxy import AsyncTLSInterceptProxy


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
                with pytest.raises(httpx.RemoteProtocolError):
                    await client.get(
                        f"https://{upstream_host}:{upstream_port}/invalid"
                    )

        assert zero_response.status_code == 200
        assert zero_response.text == ""

        first_recorded = await proxy.next_response(timeout=1.0)
        assert first_recorded.body == ""

        second_recorded = await proxy.next_response(timeout=1.0)
        assert second_recorded.body == ""
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
    assert "Example Domain" in recorded_response.body
