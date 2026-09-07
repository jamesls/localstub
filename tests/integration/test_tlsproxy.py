import asyncio
import gzip
import http.client
import logging
import socket
import ssl
import threading
import time

import httpx
import pytest
import trustme

from localstub.http.headers import Headers
from localstub.middleware import ResponderContext
from localstub.server import (
    AsyncHTTPTestServer,
    ByteFlip,
    HTTPResponse,
    TruncateBody,
)
from localstub.tlsproxy import (
    AsyncTLSInterceptProxy,
    TransformContext,
    TransformResult,
    fault_step_transformer,
)

LOG = logging.getLogger(__name__)


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
        assert request.target == "/"
        assert request.headers is not None
        assert request.headers["host"] == "example.com"


@pytest.mark.asyncio
async def test_wire_logging_emits_hexdump_when_debug_enabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG, logger="localstub.tlsproxy")
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"ok": True})

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
                response = await client.get("https://example.com/")

    assert response.status_code == 200
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "localstub.tlsproxy"
    ]
    assert any("CONNECT example.com:443" in message for message in messages)
    assert any("0x0000:" in message for message in messages)


@pytest.mark.asyncio
async def test_wire_logging_is_skipped_when_debug_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="localstub.tlsproxy")
    async with AsyncTLSInterceptProxy() as proxy:
        host, port = proxy.address
        reader, writer = await asyncio.open_connection(host, port)

        writer.write(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
        await writer.drain()

        assert b"400 Bad Request" in await reader.read(1024)

        writer.close()
        await writer.wait_closed()

    assert not [
        record
        for record in caplog.records
        if record.name == "localstub.tlsproxy"
    ]


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
        assert request.target == "/ipv6"
        assert request.headers is not None
        assert "2001:db8::1" in request.headers["host"]


def test_proxy_address_raises_before_start():
    proxy = AsyncTLSInterceptProxy()
    with pytest.raises(RuntimeError):
        _ = proxy.address


@pytest.mark.asyncio
async def test_proxy_endpoint_url_matches_bound_address() -> None:
    proxy = AsyncTLSInterceptProxy()
    with pytest.raises(RuntimeError):
        _ = proxy.endpoint_url

    async with proxy:
        host, port = proxy.address
        assert proxy.endpoint_url == f"http://{host}:{port}"


@pytest.mark.asyncio
async def test_next_request_and_response_timeout_raise():
    proxy = AsyncTLSInterceptProxy()

    with pytest.raises(asyncio.TimeoutError):
        await proxy.next_request(timeout=0.01)

    with pytest.raises(asyncio.TimeoutError):
        await proxy.next_response(timeout=0.01)

    with pytest.raises(asyncio.TimeoutError):
        await proxy.next_exchange(timeout=0.01)


def test_next_exchange_nowait_returns_none_when_no_exchange():
    proxy = AsyncTLSInterceptProxy()
    assert proxy.next_exchange_nowait() is None


@pytest.mark.asyncio
async def test_next_exchange_nowait_returns_queued_exchange():
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
                f"https://127.0.0.1:{unused_port}/queued",
                follow_redirects=False,
            )

    # The exchange is recorded before the 502 is written to the client,
    # so it is guaranteed to be queued once the response is received.
    assert response.status_code == 502

    exchange = proxy.next_exchange_nowait()
    assert exchange is not None
    assert exchange.request.target == "/queued"
    assert proxy.next_exchange_nowait() is None


@pytest.mark.asyncio
async def test_start_idempotent_and_aclose_noop():
    proxy = AsyncTLSInterceptProxy()

    async with proxy:
        first_host, first_port = proxy.address
        await proxy.start()
        assert proxy.address == (first_host, first_port)

    await proxy.aclose()


@pytest.mark.asyncio
async def test_aclose_cancels_idle_client_connection() -> None:
    proxy = AsyncTLSInterceptProxy()
    await proxy.start()
    host, port = proxy.address
    reader, writer = await asyncio.open_connection(host, port)
    await asyncio.sleep(0)

    try:
        await asyncio.wait_for(proxy.aclose(), timeout=2.0)
        eof = await asyncio.wait_for(reader.read(1), timeout=0.5)
        assert eof == b""
    finally:
        writer.close()
        await writer.wait_closed()
        await proxy.aclose()


@pytest.mark.parametrize("accept_progress_turns", [1, 2, 3, 4])
@pytest.mark.asyncio
async def test_aclose_closes_half_accepted_client_connection(
    accept_progress_turns: int,
) -> None:
    proxy = AsyncTLSInterceptProxy()
    await proxy.start()
    host, port = proxy.address
    sock = socket.create_connection((host, port))
    sock.setblocking(False)

    try:
        for _ in range(accept_progress_turns):
            await asyncio.sleep(0)

        await asyncio.wait_for(proxy.aclose(), timeout=2.0)

        loop = asyncio.get_running_loop()
        try:
            data = await asyncio.wait_for(
                loop.sock_recv(sock, 1),
                timeout=0.5,
            )
            assert data == b""
        except ConnectionResetError:
            pass
    finally:
        sock.close()
        await proxy.aclose()


@pytest.mark.parametrize("connection", ["keep-alive", "close"])
@pytest.mark.asyncio
async def test_aclose_finishes_when_tls_client_stops_reading(
    connection: str,
) -> None:
    async with (
        AsyncHTTPTestServer() as upstream,
        AsyncTLSInterceptProxy(
            default_mode="forward", upstream_tls=False
        ) as proxy,
    ):
        reader, writer = await asyncio.open_connection(*proxy.address)
        try:
            upstream_host, upstream_port = upstream.host, upstream.port
            assert upstream_host is not None
            assert upstream_port is not None
            authority = f"{upstream_host}:{upstream_port}"
            writer.write(
                f"CONNECT {authority} HTTP/1.1\r\n"
                f"Host: {authority}\r\n\r\n".encode()
            )
            await writer.drain()
            response = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=1.0
            )
            assert response.startswith(b"HTTP/1.1 200")
            context = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )
            await writer.start_tls(context, server_hostname=upstream_host)
            writer.transport.pause_reading()
            writer.write(
                f"GET / HTTP/1.1\r\nHost: {authority}\r\n"
                f"Connection: {connection}\r\n\r\n".encode()
            )
            await writer.drain()
            await upstream.next_request(timeout=1.0)

            await asyncio.wait_for(proxy.aclose(), timeout=2.0)
        finally:
            writer.transport.resume_reading()
            writer.close()
            await writer.wait_closed()


@pytest.mark.asyncio
async def test_aclose_cancels_forward_connection_after_tls_upgrade() -> None:
    request_received = asyncio.Event()

    async def withhold_response(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            request_received.set()
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()

    upstream = await asyncio.start_server(
        withhold_response,
        "127.0.0.1",
        0,
    )
    upstream_host, upstream_port = upstream.sockets[0].getsockname()[:2]
    proxy = AsyncTLSInterceptProxy(
        default_mode="forward",
        upstream_tls=False,
    )
    await proxy.start()
    proxy_host, proxy_port = proxy.address
    verify_ctx = ssl.create_default_context(cafile=str(proxy.ca.ca_pem_path()))

    try:
        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            request_task = asyncio.create_task(
                client.get(f"https://{upstream_host}:{upstream_port}/pending")
            )
            await asyncio.wait_for(request_received.wait(), timeout=1.0)
            await asyncio.wait_for(proxy.aclose(), timeout=2.0)

            with pytest.raises(httpx.TransportError):
                await request_task
    finally:
        await proxy.aclose()
        upstream.close()
        await upstream.wait_closed()


@pytest.mark.asyncio
async def test_aclose_called_from_request_handler_completes() -> None:
    aclose_finished = asyncio.Event()

    async with AsyncHTTPTestServer() as server:
        proxy = AsyncTLSInterceptProxy(server=server)
        await proxy.start()

        async def shutdown_handler(_: ResponderContext) -> HTTPResponse:
            await proxy.aclose()
            aclose_finished.set()
            return HTTPResponse.text("closing")

        server.handler = shutdown_handler
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        try:
            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                # Shutdown tears down the connection before the response
                # is written, so the client sees a transport failure.
                with pytest.raises(httpx.TransportError):
                    await asyncio.wait_for(
                        client.get("https://example.com/"),
                        timeout=5.0,
                    )
            await asyncio.wait_for(aclose_finished.wait(), timeout=2.0)
        finally:
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
    assert recorded_request.target == "/unreachable"
    assert recorded_request.headers is not None
    assert "127.0.0.1" in recorded_request.headers["host"]

    # No upstream response should be recorded.
    with pytest.raises(asyncio.TimeoutError):
        await proxy.next_response(timeout=0.1)

    exchange = await proxy.next_exchange(timeout=1.0)
    assert exchange.request is recorded_request
    assert exchange.response is None


@pytest.mark.asyncio
async def test_forward_returns_502_on_connect_fail_with_expect_100_continue():
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
        ca_pem_path = str(proxy.ca.ca_pem_path())

        try:
            response_bytes = await asyncio.wait_for(
                asyncio.to_thread(
                    _do_100_continue_headers_only_request,
                    proxy_host,
                    proxy_port,
                    "127.0.0.1",
                    unused_port,
                    ca_pem_path,
                ),
                timeout=10.0,
            )
        except TimeoutError:
            pytest.fail(
                "Timed out waiting for 502 after sending Expect: 100-continue "
                "headers. The proxy likely blocked while reading a body that "
                "the client never sent."
            )

    assert b"502" in response_bytes

    recorded_request = await proxy.next_request(timeout=1.0)
    assert recorded_request.method == "PUT"
    assert recorded_request.target == "/unreachable-continue"
    assert recorded_request.headers is not None
    assert recorded_request.headers["expect"] == "100-continue"
    assert recorded_request.body == b""

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
        assert recorded.body == b"peekboo"
        assert b"X-Trail: done" in recorded.wire_raw_bytes
        exchange = await proxy.next_exchange(timeout=1.0)
        assert exchange.request.target == "/chunked"
        assert exchange.response is recorded
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
        assert recorded.body == b"hello gzip"
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
        assert recorded.body == b"not gzipped"
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
        assert first_recorded.body == b""

        with pytest.raises(asyncio.TimeoutError):
            await proxy.next_response(timeout=0.1)
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_to_plain_http_when_tls_disabled():

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
        assert recorded.target == "/forward"
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
        assert recorded_response.body == b"tls-ok"
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
        assert recorded_response.body == b"streamed close body"
    finally:
        server.close()
        await server.wait_closed()


def _send_requests_over_one_tunnel(
    proxy_host: str,
    proxy_port: int,
    upstream_host: str,
    upstream_port: int,
    ca_pem_path: str,
    count: int,
) -> list[tuple[int, bytes]]:
    """Send *count* requests over a single tunneled connection.

    Mirrors a real client (http.client) that establishes one CONNECT tunnel
    and reuses it for sequential requests.
    """
    ctx = ssl.create_default_context(cafile=ca_pem_path)
    conn = http.client.HTTPSConnection(
        proxy_host, proxy_port, context=ctx, timeout=10.0
    )
    conn.set_tunnel(upstream_host, upstream_port)
    results: list[tuple[int, bytes]] = []
    try:
        for _ in range(count):
            conn.request("GET", "/")
            response = conn.getresponse()
            results.append((response.status, response.read()))
    finally:
        conn.close()
    return results


@pytest.mark.asyncio
async def test_forward_reuses_tunnel_for_multiple_requests():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 5\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\nhello"
        )
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            LOG.debug("Failed to close test writer", exc_info=True)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            results = await asyncio.wait_for(
                asyncio.to_thread(
                    _send_requests_over_one_tunnel,
                    proxy_host,
                    proxy_port,
                    upstream_host,
                    upstream_port,
                    str(proxy.ca.ca_pem_path()),
                    2,
                ),
                timeout=10.0,
            )

        assert results == [(200, b"hello"), (200, b"hello")]

        first = await proxy.next_exchange(timeout=1.0)
        second = await proxy.next_exchange(timeout=1.0)
        assert first.request.target == "/"
        assert second.request.target == "/"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_closes_tunnel_when_client_requests_connection_close():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nbye")
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            LOG.debug("Failed to close test writer", exc_info=True)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await asyncio.open_connection(*proxy.address)
            try:
                writer.write(
                    f"CONNECT {upstream_host}:{upstream_port} HTTP/1.1\r\n"
                    f"Host: {upstream_host}:{upstream_port}\r\n"
                    "\r\n".encode()
                )
                await writer.drain()
                connect_response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert connect_response.startswith(b"HTTP/1.1 200")

                verify_context = ssl.create_default_context(
                    cafile=str(proxy.ca.ca_pem_path())
                )
                await writer.start_tls(
                    verify_context, server_hostname=upstream_host
                )

                writer.write(
                    b"GET / HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n".encode()
                    + b"Connection: close\r\n\r\n"
                )
                await writer.drain()

                headers = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert headers.startswith(b"HTTP/1.1 200")
                body = await asyncio.wait_for(
                    reader.readexactly(3), timeout=1.0
                )
                assert body == b"bye"

                # The client asked to close, so the proxy must not keep the
                # tunnel alive: the next read observes EOF (or a reset).
                try:
                    eof = await asyncio.wait_for(reader.read(1), timeout=1.0)
                    assert eof == b""
                except (ConnectionResetError, ssl.SSLError):
                    pass
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_returns_400_for_malformed_request_over_tunnel():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # A malformed client request is rejected before any upstream
        # connection is opened, so this handler is never invoked.
        await reader.read()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await asyncio.open_connection(*proxy.address)
            try:
                writer.write(
                    f"CONNECT {upstream_host}:{upstream_port} HTTP/1.1\r\n"
                    f"Host: {upstream_host}:{upstream_port}\r\n"
                    "\r\n".encode()
                )
                await writer.drain()
                connect_response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert connect_response.startswith(b"HTTP/1.1 200")

                verify_context = ssl.create_default_context(
                    cafile=str(proxy.ca.ca_pem_path())
                )
                await writer.start_tls(
                    verify_context, server_hostname=upstream_host
                )

                writer.write(b"INVALID HTTP DATA\r\n\r\n")
                await writer.drain()

                response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert response.startswith(b"HTTP/1.1 400")
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


async def _establish_forward_tls_tunnel(
    proxy: AsyncTLSInterceptProxy,
    upstream_host: str,
    upstream_port: int,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a CONNECT tunnel to *upstream* and upgrade the client to TLS."""
    reader, writer = await asyncio.open_connection(*proxy.address)
    writer.write(
        f"CONNECT {upstream_host}:{upstream_port} HTTP/1.1\r\n"
        f"Host: {upstream_host}:{upstream_port}\r\n"
        "\r\n".encode()
    )
    await writer.drain()
    connect_response = await asyncio.wait_for(
        reader.readuntil(b"\r\n\r\n"), timeout=1.0
    )
    assert connect_response.startswith(b"HTTP/1.1 200")
    verify_context = ssl.create_default_context(
        cafile=str(proxy.ca.ca_pem_path())
    )
    await writer.start_tls(verify_context, server_hostname=upstream_host)
    return reader, writer


@pytest.mark.asyncio
async def test_forward_returns_400_when_request_body_malformed():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        # The malformed body is rejected by the proxy, so a full request
        # never reaches upstream.
        await reader.read()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                writer.write(
                    b"POST / HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n".encode()
                    + b"Transfer-Encoding: chunked\r\n\r\n"
                    + b"XYZ\r\n"
                )
                await writer.drain()

                response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert response.startswith(b"HTTP/1.1 400")
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_closes_tunnel_after_early_final_expect_response():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 417 Expectation Failed\r\nContent-Length: 0\r\n\r\n"
        )
        await writer.drain()
        await reader.read()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                writer.write(
                    b"PUT /upload HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n".encode()
                    + b"Content-Length: 18\r\n"
                    + b"Expect: 100-continue\r\n\r\n"
                )
                await writer.drain()

                response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert response.startswith(b"HTTP/1.1 417")

                # A client that stopped waiting for the interim response
                # may already have committed its body to the wire.  The
                # proxy never read that body, so it must close the tunnel
                # instead of parsing the body as another request.
                try:
                    writer.write(b"late body arriving")
                    await writer.drain()
                    trailing = await asyncio.wait_for(
                        reader.read(4096), timeout=1.0
                    )
                    assert trailing == b""
                except (ConnectionResetError, ssl.SSLError):
                    pass
            finally:
                writer.close()
                await writer.wait_closed()

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 417
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_closes_tunnel_after_101_switching_protocols():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n\r\n"
        )
        await writer.drain()
        await reader.read()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                writer.write(
                    b"GET /ws HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n".encode()
                    + b"Upgrade: websocket\r\n"
                    + b"Connection: Upgrade\r\n\r\n"
                )
                await writer.drain()

                response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert response.startswith(b"HTTP/1.1 101")

                # After a protocol switch the tunnel no longer carries
                # HTTP.  Without a bidirectional relay the proxy must
                # close rather than parse frames as another request.
                try:
                    writer.write(b"\x81\x05hello")
                    await writer.drain()
                    trailing = await asyncio.wait_for(
                        reader.read(4096), timeout=1.0
                    )
                    assert trailing == b""
                except (ConnectionResetError, ssl.SSLError):
                    pass
            finally:
                writer.close()
                await writer.wait_closed()

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 101
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_returns_400_when_100_continue_body_malformed():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
        await reader.read()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                writer.write(
                    b"PUT / HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n".encode()
                    + b"Transfer-Encoding: chunked\r\n"
                    + b"Expect: 100-continue\r\n\r\n"
                )
                await writer.drain()

                continue_response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert continue_response == b"HTTP/1.1 100 Continue\r\n\r\n"

                writer.write(b"XYZ\r\n")
                await writer.drain()

                final = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert final.startswith(b"HTTP/1.1 400")
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_returns_502_when_100_continue_response_unparseable():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
        await reader.readexactly(4)
        writer.write(b"GARBAGE NOT HTTP\r\n\r\n")
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                writer.write(
                    b"PUT / HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n".encode()
                    + b"Content-Length: 4\r\n"
                    + b"Expect: 100-continue\r\n\r\n"
                )
                await writer.drain()

                continue_response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert continue_response == b"HTTP/1.1 100 Continue\r\n\r\n"

                writer.write(b"data")
                await writer.drain()

                final = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert final.startswith(b"HTTP/1.1 502")
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


async def _expect_tunnel_eof(reader: asyncio.StreamReader) -> None:
    """Assert the proxy closed the tunnel (EOF or reset) after a response."""
    try:
        trailing = await asyncio.wait_for(reader.read(1), timeout=1.0)
        assert trailing == b""
    except (ConnectionResetError, ssl.SSLError):
        pass


@pytest.mark.asyncio
async def test_forward_closes_tunnel_after_http10_response_without_keepalive():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.0 200 OK\r\nContent-Length: 5\r\n\r\nhello")
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            LOG.debug("Failed to close test writer", exc_info=True)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                writer.write(
                    b"GET / HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n\r\n".encode()
                )
                await writer.drain()

                headers = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert headers.startswith(b"HTTP/1.0 200")
                body = await asyncio.wait_for(
                    reader.readexactly(5), timeout=1.0
                )
                assert body == b"hello"

                # Persistence is opt-in for HTTP/1.0 responses; without
                # Connection: keep-alive the proxy must close the tunnel.
                await _expect_tunnel_eof(reader)
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_reuses_tunnel_after_http10_keep_alive_response():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.0 200 OK\r\n"
            b"Content-Length: 5\r\n"
            b"Connection: keep-alive\r\n"
            b"\r\nhello"
        )
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            LOG.debug("Failed to close test writer", exc_info=True)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                for _ in range(2):
                    writer.write(
                        b"GET / HTTP/1.1\r\n"
                        + f"Host: {upstream_host}:{upstream_port}\r\n".encode()
                        + b"\r\n"
                    )
                    await writer.drain()

                    headers = await asyncio.wait_for(
                        reader.readuntil(b"\r\n\r\n"), timeout=1.0
                    )
                    assert headers.startswith(b"HTTP/1.0 200")
                    body = await asyncio.wait_for(
                        reader.readexactly(5), timeout=1.0
                    )
                    assert body == b"hello"
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_closes_tunnel_when_transformer_override_says_close():
    def close_transformer(upstream: TransformContext) -> TransformResult:
        return TransformResult(
            override_response=HTTPResponse.text(
                "goodbye", headers={"Connection": "close"}
            )
        )

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            LOG.debug("Failed to close test writer", exc_info=True)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=close_transformer,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                writer.write(
                    b"GET / HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n\r\n".encode()
                )
                await writer.drain()

                headers = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert headers.startswith(b"HTTP/1.1 200")
                assert b"Connection: close" in headers
                body = await asyncio.wait_for(
                    reader.readexactly(7), timeout=1.0
                )
                assert body == b"goodbye"

                # The client was told Connection: close, so the proxy
                # must close even though the upstream response did not
                # ask for it.
                await _expect_tunnel_eof(reader)
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_closes_expect_tunnel_after_transformed_101() -> None:
    def switching_transformer(
        _upstream: TransformContext,
    ) -> TransformResult:
        return TransformResult(
            override_response=HTTPResponse.raw(
                b"",
                status=101,
                headers={
                    "Upgrade": "websocket",
                    "Connection": "Upgrade",
                },
            )
        )

    async with (
        AsyncHTTPTestServer(
            default_response=HTTPResponse.raw(b""),
        ) as upstream,
        AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=switching_transformer,
        ) as proxy,
    ):
        assert upstream.host is not None
        assert upstream.port is not None
        reader, writer = await _establish_forward_tls_tunnel(
            proxy, upstream.host, upstream.port
        )
        try:
            writer.write(
                b"GET /ws HTTP/1.1\r\n"
                + f"Host: {upstream.host}:{upstream.port}\r\n".encode()
                + b"Expect: 100-continue\r\n"
                + b"Upgrade: websocket\r\n"
                + b"Connection: Upgrade\r\n\r\n"
            )
            await writer.drain()

            response = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"), timeout=1.0
            )
            assert response.startswith(b"HTTP/1.1 101")
            await _expect_tunnel_eof(reader)
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.mark.asyncio
async def test_forward_reuses_tunnel_when_transformer_rewrites_body():
    def uppercase_transformer(upstream: TransformContext) -> TransformResult:
        return TransformResult(body=upstream.body.upper())

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"X-Obs: \x80\xff\r\n"
            b"Content-Length: 5\r\n\r\n"
            b"hello"
        )
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            LOG.debug("Failed to close test writer", exc_info=True)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=uppercase_transformer,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                for _ in range(2):
                    writer.write(
                        b"GET / HTTP/1.1\r\n"
                        + f"Host: {upstream_host}:{upstream_port}\r\n".encode()
                        + b"\r\n"
                    )
                    await writer.drain()

                    headers = await asyncio.wait_for(
                        reader.readuntil(b"\r\n\r\n"), timeout=1.0
                    )
                    assert headers.startswith(b"HTTP/1.1 200")
                    assert b"X-Obs: \x80\xff\r\n" in headers
                    body = await asyncio.wait_for(
                        reader.readexactly(5), timeout=1.0
                    )
                    assert body == b"HELLO"
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_reuses_tunnel_after_bodyless_expect_request():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        await writer.drain()
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            LOG.debug("Failed to close test writer", exc_info=True)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await _establish_forward_tls_tunnel(
                proxy, upstream_host, upstream_port
            )
            try:
                # A bodyless request with Expect: 100-continue that gets
                # an immediate final response leaves nothing unread, so
                # the tunnel stays reusable.
                writer.write(
                    b"GET / HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n".encode()
                    + b"Expect: 100-continue\r\n\r\n"
                )
                await writer.drain()

                headers = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert headers.startswith(b"HTTP/1.1 200")
                body = await asyncio.wait_for(
                    reader.readexactly(2), timeout=1.0
                )
                assert body == b"ok"

                writer.write(
                    b"GET / HTTP/1.1\r\n"
                    + f"Host: {upstream_host}:{upstream_port}\r\n\r\n".encode()
                )
                await writer.drain()

                headers = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"), timeout=1.0
                )
                assert headers.startswith(b"HTTP/1.1 200")
                body = await asyncio.wait_for(
                    reader.readexactly(2), timeout=1.0
                )
                assert body == b"ok"
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_proxy_records_request_and_response_when_forwarding_to_real():

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
    assert recorded_request.target == "/"
    assert recorded_request.headers is not None
    assert recorded_request.headers["host"] == "example.com"

    recorded_response = await proxy.next_response(timeout=2.0)
    assert recorded_response.status == 200
    assert recorded_response.body is not None
    assert b"Example Domain" in recorded_response.body


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
        assert recorded_request.target == "/authors.html"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        # The recorded response body should be empty for HEAD
        assert recorded_response.body == b""
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_handles_client_closing_connection_early():

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
            LOG.debug("Failed to close test writer", exc_info=True)

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
        assert recorded_request.target == "/test"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        assert recorded_response.body == b"hello"
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
        LOG.debug("Failed to close test writer", exc_info=True)


@pytest.mark.asyncio
async def test_forward_handles_100_continue_before_final_response():
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
        assert recorded_request.target == "/upload"

        # Only the final response should be recorded
        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        assert recorded_response.body == b"upload accepted"
    finally:
        server.close()
        await server.wait_closed()


async def _early_hints_then_continue_handler(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    try:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 103 Early Hints\r\n"
            b"Link: </style.css>; rel=preload\r\n"
            b"\r\n"
            b"HTTP/1.1 100 Continue\r\n"
            b"\r\n"
        )
        await writer.drain()
        body = await reader.readexactly(4)
        assert body == b"data"
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 2\r\n"
            b"Connection: close\r\n"
            b"\r\n"
            b"ok"
        )
        await writer.drain()
    except asyncio.IncompleteReadError:
        pass
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_forward_relays_early_hints_before_100_continue() -> None:
    server = await asyncio.start_server(
        _early_hints_then_continue_handler,
        "127.0.0.1",
        0,
    )
    assert server.sockets is not None
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            reader, writer = await asyncio.open_connection(*proxy.address)
            try:
                writer.write(
                    f"CONNECT {server_host}:{server_port} HTTP/1.1\r\n"
                    f"Host: {server_host}:{server_port}\r\n"
                    "\r\n".encode()
                )
                await writer.drain()
                connect_response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"),
                    timeout=0.5,
                )
                assert connect_response.startswith(b"HTTP/1.1 200")

                verify_context = ssl.create_default_context(
                    cafile=str(proxy.ca.ca_pem_path())
                )
                await writer.start_tls(
                    verify_context,
                    server_hostname=server_host,
                )

                writer.write(
                    b"PUT /upload HTTP/1.1\r\n"
                    + f"Host: {server_host}:{server_port}\r\n".encode()
                    + b"Content-Length: 4\r\n"
                    + b"Expect: 100-continue\r\n"
                    + b"Connection: close\r\n"
                    + b"\r\n"
                )
                await writer.drain()

                early_hints = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"),
                    timeout=0.5,
                )
                assert early_hints.startswith(b"HTTP/1.1 103")
                continue_response = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"),
                    timeout=0.5,
                )
                assert continue_response == b"HTTP/1.1 100 Continue\r\n\r\n"

                writer.write(b"data")
                await writer.drain()
                final_headers = await asyncio.wait_for(
                    reader.readuntil(b"\r\n\r\n"),
                    timeout=0.5,
                )
                final_body = await asyncio.wait_for(
                    reader.readexactly(2),
                    timeout=0.5,
                )
                assert final_headers.startswith(b"HTTP/1.1 200")
                assert final_body == b"ok"
            finally:
                writer.close()
                await writer.wait_closed()
    finally:
        server.close()
        await server.wait_closed()


async def _proper_expect_continue_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """
    Handler that properly implements 100-continue semantics:
    1. Read ONLY headers
    2. If Expect: 100-continue, send 100 Continue
    3. THEN wait for and read the body
    4. Send final response
    """
    # Read only headers
    headers_data = await reader.readuntil(b"\r\n\r\n")

    # Parse Content-Length from headers
    headers_str = headers_data.decode("utf-8", errors="replace")
    content_length = 0
    for line in headers_str.split("\r\n"):
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":")[1].strip())
            break

    # Send 100 Continue
    writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
    await writer.drain()

    # Now wait for and read the body
    body = b""
    if content_length > 0:
        body = await reader.readexactly(content_length)

    # Send final response
    response_body = f"received {len(body)} bytes".encode()
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: " + str(len(response_body)).encode() + b"\r\n"
        b"Content-Type: text/plain\r\n"
        b"\r\n" + response_body
    )
    await writer.drain()
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        LOG.debug("Failed to close test writer", exc_info=True)


def _do_100_continue_request(
    proxy_host: str,
    proxy_port: int,
    upstream_host: str,
    upstream_port: int,
    ca_pem_path: str,
    *,
    path: str = "/upload",
    body_content: bytes = b"test body data for 100-continue",
) -> tuple[bytes, bytes]:
    """
    Execute HTTP request with proper 100-continue semantics using sync sockets.

    Returns tuple of (100-continue response, final response).
    """
    sock = socket.create_connection((proxy_host, proxy_port), timeout=5.0)
    try:
        sock.settimeout(None)

        # Send CONNECT and get tunnel
        connect_req = (
            f"CONNECT {upstream_host}:{upstream_port} HTTP/1.1\r\n"
            f"Host: {upstream_host}:{upstream_port}\r\n"
            "\r\n"
        ).encode()
        sock.sendall(connect_req)

        connect_resp, _ = _recv_until_with_timeout(
            sock,
            b"\r\n\r\n",
            timeout=3.0,
            timeout_message="Timeout waiting for CONNECT response from proxy.",
        )
        if b"200" not in connect_resp:
            raise AssertionError(
                f"Expected 200 from CONNECT: {connect_resp!r}"
            )

        # Upgrade to TLS and trust the proxy's CA
        sock.settimeout(3.0)
        ssl_ctx = ssl.create_default_context(cafile=ca_pem_path)
        tls_sock: ssl.SSLSocket | None = None
        tls_sock = ssl_ctx.wrap_socket(sock, server_hostname=upstream_host)
        try:
            tls_sock.settimeout(None)

            # Send ONLY headers with Expect: 100-continue
            request_headers = (
                f"PUT {path} HTTP/1.1\r\n"
                f"Host: {upstream_host}:{upstream_port}\r\n"
                f"Content-Length: {len(body_content)}\r\n"
                f"Expect: 100-continue\r\n"
                f"\r\n"
            ).encode()
            tls_sock.sendall(request_headers)

            # Wait for 100 Continue (this is where the bug manifests)
            first_response, tail = _recv_until_with_timeout(
                tls_sock,
                b"\r\n\r\n",
                timeout=3.0,
                timeout_message=(
                    "Timeout waiting for 100 Continue response. "
                    "The proxy is likely stuck reading the full request "
                    "before forwarding headers to upstream."
                ),
            )
            if b"100" not in first_response:
                raise AssertionError(
                    f"Expected 100 Continue, got: {first_response!r}"
                )

            # Now send body
            if body_content:
                tls_sock.sendall(body_content)

            # Read final response
            final_response = _recv_http_response_from_prefetched(
                tls_sock,
                tail,
                timeout_message="Timeout waiting for final response.",
            )
            return first_response, final_response
        finally:
            if tls_sock is not None:
                tls_sock.close()
    finally:
        sock.close()


def _do_100_continue_headers_only_request(
    proxy_host: str,
    proxy_port: int,
    upstream_host: str,
    upstream_port: int,
    ca_pem_path: str,
) -> bytes:
    """Send Expect: 100-continue headers but withhold the body.

    This simulates a correct 100-continue client that waits for an interim
    response before sending the request body.
    """
    sock = socket.create_connection((proxy_host, proxy_port), timeout=5.0)
    try:
        sock.settimeout(None)

        connect_req = (
            f"CONNECT {upstream_host}:{upstream_port} HTTP/1.1\r\n"
            f"Host: {upstream_host}:{upstream_port}\r\n"
            "\r\n"
        ).encode()
        sock.sendall(connect_req)

        connect_resp, _ = _recv_until_with_timeout(
            sock,
            b"\r\n\r\n",
            timeout=3.0,
            timeout_message="Timeout waiting for CONNECT response from proxy.",
        )
        if b"200" not in connect_resp:
            raise AssertionError(
                f"Expected 200 from CONNECT: {connect_resp!r}"
            )

        sock.settimeout(3.0)
        ssl_ctx = ssl.create_default_context(cafile=ca_pem_path)
        tls_sock: ssl.SSLSocket | None = None
        tls_sock = ssl_ctx.wrap_socket(sock, server_hostname=upstream_host)
        try:
            tls_sock.settimeout(None)

            body_content = b"test body data for 100-continue"
            request_headers = (
                f"PUT /unreachable-continue HTTP/1.1\r\n"
                f"Host: {upstream_host}:{upstream_port}\r\n"
                f"Content-Length: {len(body_content)}\r\n"
                f"Expect: 100-continue\r\n"
                f"\r\n"
            ).encode()
            tls_sock.sendall(request_headers)

            response, _ = _recv_until_with_timeout(
                tls_sock,
                b"\r\n\r\n",
                timeout=3.0,
                timeout_message=(
                    "Timeout waiting for final response after sending only "
                    "Expect: 100-continue headers."
                ),
            )
            return response
        finally:
            if tls_sock is not None:
                tls_sock.close()
    finally:
        sock.close()


def _recv_until(
    sock: socket.socket,
    delimiter: bytes,
) -> tuple[bytes, bytes]:
    """Receive data until delimiter is found, returning (head, tail)."""
    data = b""
    while delimiter not in data:
        chunk = sock.recv(1024)
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    head, tail = data.split(delimiter, 1)
    return head + delimiter, tail


def _recv_until_with_timeout(
    sock: socket.socket,
    delimiter: bytes,
    *,
    timeout: float,
    timeout_message: str,
) -> tuple[bytes, bytes]:
    timer = threading.Timer(timeout, sock.close)
    timer.daemon = True
    timer.start()
    try:
        return _recv_until(sock, delimiter)
    except Exception as exc:
        raise AssertionError(timeout_message) from exc
    finally:
        timer.cancel()


def _recv_exactly(
    sock: ssl.SSLSocket,
    size: int,
) -> bytes:
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("Connection closed")
        data += chunk
    return data


def _recv_exactly_with_timeout(
    sock: ssl.SSLSocket,
    size: int,
    *,
    timeout: float,
    timeout_message: str,
) -> bytes:
    timer = threading.Timer(timeout, sock.close)
    timer.daemon = True
    timer.start()
    try:
        return _recv_exactly(sock, size)
    except Exception as exc:
        raise AssertionError(timeout_message) from exc
    finally:
        timer.cancel()


def _parse_content_length(headers: bytes) -> int | None:
    for line in headers.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            value = line.split(b":", 1)[1].strip()
            return int(value)
    return None


def _recv_http_response_from_prefetched(
    sock: ssl.SSLSocket,
    prefetched: bytes,
    *,
    timeout_message: str,
) -> bytes:
    """Receive a full HTTP response, starting with prefetched bytes."""
    timer = threading.Timer(3.0, sock.close)
    timer.daemon = True
    timer.start()
    try:
        data = prefetched
        while b"\r\n\r\n" not in data:
            chunk = sock.recv(1024)
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk

        headers, body_start = data.split(b"\r\n\r\n", 1)
        headers += b"\r\n\r\n"
        content_length = _parse_content_length(headers)
        if content_length is None:
            return headers + body_start
        if len(body_start) < content_length:
            body_start += _recv_exactly_with_timeout(
                sock,
                content_length - len(body_start),
                timeout=3.0,
                timeout_message=timeout_message,
            )
        return headers + body_start[:content_length]
    except Exception as exc:
        raise AssertionError(timeout_message) from exc
    finally:
        timer.cancel()


def _recv_http_response(
    sock: ssl.SSLSocket,
    *,
    timeout_message: str,
) -> bytes:
    headers, body_start = _recv_until_with_timeout(
        sock,
        b"\r\n\r\n",
        timeout=3.0,
        timeout_message=timeout_message,
    )
    content_length = _parse_content_length(headers)
    if content_length is None:
        return headers + body_start
    if len(body_start) < content_length:
        body_start += _recv_exactly_with_timeout(
            sock,
            content_length - len(body_start),
            timeout=3.0,
            timeout_message=timeout_message,
        )
    return headers + body_start[:content_length]


@pytest.mark.asyncio
async def test_forward_client_waits_for_100_continue_before_sending_body():
    upstream_server = await asyncio.start_server(
        _proper_expect_continue_handler,
        "127.0.0.1",
        0,
    )
    upstream_host, upstream_port = upstream_server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            ca_pem_path = str(proxy.ca.ca_pem_path())

            try:
                first_resp, final_resp = await asyncio.wait_for(
                    asyncio.to_thread(
                        _do_100_continue_request,
                        proxy_host,
                        proxy_port,
                        upstream_host,
                        upstream_port,
                        ca_pem_path,
                    ),
                    timeout=10.0,
                )
            except TimeoutError:
                pytest.fail(
                    "Test timed out waiting for 100-continue flow. "
                    "The proxy likely has a bug in forward mode where it "
                    "doesn't relay 100-continue from upstream to client."
                )

            assert b"100" in first_resp
            assert b"200 OK" in final_resp
            assert b"received 31 bytes" in final_resp

            recorded_request = await proxy.next_request(timeout=1.0)
            assert recorded_request.method == "PUT"
            assert recorded_request.target == "/upload"

    finally:
        upstream_server.close()
        await upstream_server.wait_closed()


async def _coalesced_expect_continue_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Handler that sends 100 Continue and 200 OK in one write."""
    body = b"OK"
    writer.write(
        b"HTTP/1.1 100 Continue\r\n\r\n"
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 2\r\n"
        b"Content-Type: text/plain\r\n"
        b"\r\n" + body
    )
    await writer.drain()

    # Keep the connection open long enough for the proxy to send the request.
    try:
        await reader.readuntil(b"\r\n\r\n")
    except Exception:
        LOG.debug("Failed to read test request", exc_info=True)

    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        LOG.debug("Failed to close test writer", exc_info=True)


@pytest.mark.asyncio
async def test_forward_preserves_final_response_when_100_and_200_coalesce():
    upstream_server = await asyncio.start_server(
        _coalesced_expect_continue_handler,
        "127.0.0.1",
        0,
    )
    upstream_host, upstream_port = upstream_server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            ca_pem_path = str(proxy.ca.ca_pem_path())

            try:
                first_resp, final_resp = await asyncio.wait_for(
                    asyncio.to_thread(
                        _do_100_continue_request,
                        proxy_host,
                        proxy_port,
                        upstream_host,
                        upstream_port,
                        ca_pem_path,
                        path="/coalesced",
                        body_content=b"",
                    ),
                    timeout=10.0,
                )
            except TimeoutError:
                pytest.fail(
                    "Timed out waiting for final response after receiving "
                    "100 Continue. The proxy may have dropped buffered "
                    "upstream bytes."
                )

        assert b"100" in first_resp
        assert b"200 OK" in final_resp
        assert final_resp.endswith(b"OK")

        recorded_request = await proxy.next_request(timeout=1.0)
        assert recorded_request.method == "PUT"
        assert recorded_request.target == "/coalesced"
        assert recorded_request.body == b""

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        assert recorded_response.body == b"OK"
    finally:
        upstream_server.close()
        await upstream_server.wait_closed()


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
        LOG.debug("Failed to close test writer", exc_info=True)


@pytest.mark.asyncio
async def test_forward_handles_multiple_informational_responses():
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
        assert recorded_request.target == "/long-operation"

        # Only the final response should be recorded
        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 201
        assert recorded_response.body == b"done"
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
        LOG.debug("Failed to close test writer", exc_info=True)


@pytest.mark.asyncio
async def test_forward_handles_100_continue_with_chunked_response():
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
        assert recorded_request.target == "/bucket/key"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        assert recorded_response.body == b"<Success/>\n"
    finally:
        server.close()
        await server.wait_closed()


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
        LOG.debug("Failed to close test writer", exc_info=True)


@pytest.mark.asyncio
async def test_forward_transforms_response_body_with_byteflip():
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
        assert recorded.body == b'{"message":"hello","count":42}'
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_transformer_delay_before():

    def delay_transformer(upstream: TransformContext) -> TransformResult:
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

    def override_transformer(upstream: TransformContext) -> TransformResult:
        return TransformResult(
            override_response=HTTPResponse.text(
                "Service Unavailable",
                status=503,
                headers=Headers.from_raw_items([
                    (b"Content-Type", b"text/plain"),
                    (b"X-Obs", b"\x80\xff"),
                ]),
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
        assert any(
            name.lower() == b"x-obs" and value == b"\x80\xff"
            for name, value in response.headers.raw
        )

        # The recorded response should contain the original upstream response
        recorded = await proxy.next_response(timeout=1.0)
        assert recorded.status == 200
        assert recorded.body == b'{"message":"hello","count":42}'
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_async_transformer():

    async def async_transformer(upstream: TransformContext) -> TransformResult:
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

    def passthrough_transformer(upstream: TransformContext) -> TransformResult:
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

    def conditional_transformer(upstream: TransformContext) -> TransformResult:
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


async def _fixed_ok_handler(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    await reader.readuntil(b"\r\n\r\n")
    writer.write(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Length: 2\r\n"
        b"Content-Type: text/plain\r\n"
        b"Connection: close\r\n"
        b"\r\n"
        b"ok"
    )
    await writer.drain()
    writer.close()
    await writer.wait_closed()


@pytest.mark.asyncio
async def test_forward_recording_buffer_evicts_oldest_when_full():
    server = await asyncio.start_server(_fixed_ok_handler, "127.0.0.1", 0)
    upstream_host, upstream_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            recording_buffer_size=2,
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
                for i in range(3):
                    response = await client.get(
                        f"https://{upstream_host}:{upstream_port}/path-{i}"
                    )
                    assert response.status_code == 200

        assert proxy.dropped_requests == 1
        assert proxy.dropped_responses == 1
        assert proxy.dropped_exchanges == 1

        oldest = await proxy.next_request(timeout=1.0)
        assert oldest.target == "/path-1"
        exchange = await proxy.next_exchange(timeout=1.0)
        assert exchange.request.target == "/path-1"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_recording_within_buffer_never_drops():
    server = await asyncio.start_server(_fixed_ok_handler, "127.0.0.1", 0)
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
                    f"https://{upstream_host}:{upstream_port}/solo"
                )
                assert response.status_code == 200

        assert not proxy.dropped_requests
        assert not proxy.dropped_responses
        assert not proxy.dropped_exchanges
    finally:
        server.close()
        await server.wait_closed()


def test_zero_recording_buffer_size_raises_value_error():
    with pytest.raises(ValueError, match="at least 1"):
        AsyncTLSInterceptProxy(recording_buffer_size=0)
