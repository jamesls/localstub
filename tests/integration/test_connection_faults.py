from __future__ import annotations

import asyncio
import contextlib
import sys
import time

import aiohttp
import httpx
import pytest

from localstub import (
    AsyncHTTPTestServer,
    CloseConnection,
    CloseDuringRequest,
    DropConnection,
    FaultyTransmission,
    HeaderContext,
    HeaderDecision,
    HeaderNext,
    HTTPResponse,
    ResponderContext,
    close_during_request,
)
from localstub.forward import RawForwarder
from localstub.tlsproxy import fault_step_transformer

TIMEOUT = 2.0
GET_REQUEST = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
UPLOAD_HEAD = (
    b"PUT /upload HTTP/1.1\r\nHost: localhost\r\nContent-Length: 20\r\n\r\n"
)
CHUNKED_HEAD = (
    b"PUT /upload HTTP/1.1\r\n"
    b"Host: localhost\r\n"
    b"Transfer-Encoding: chunked\r\n\r\n"
)
RESPONSE_BODY = b"0123456789" * 4

reset_observable = pytest.mark.skipif(
    sys.platform == "win32",
    reason="TCP reset observation is best effort on Windows",
)


async def _connect(
    server: AsyncHTTPTestServer,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    assert server.host is not None
    assert server.port is not None
    return await asyncio.open_connection(server.host, server.port)


async def _send(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(data)
    await asyncio.wait_for(writer.drain(), timeout=TIMEOUT)


async def _close(writer: asyncio.StreamWriter) -> None:
    writer.close()
    with contextlib.suppress(OSError):
        await asyncio.wait_for(writer.wait_closed(), timeout=TIMEOUT)


async def _read_to_end(reader: asyncio.StreamReader) -> tuple[bytes, bool]:
    # Everything readable before the server closed, and whether the
    # close was observed as a reset rather than EOF.
    chunks: list[bytes] = []
    while True:
        try:
            data = await asyncio.wait_for(reader.read(4096), timeout=TIMEOUT)
        except ConnectionResetError:
            return b"".join(chunks), True
        if not data:
            return b"".join(chunks), False
        chunks.append(data)


async def _exchange_raw(
    server: AsyncHTTPTestServer, request: bytes
) -> tuple[bytes, bool]:
    reader, writer = await _connect(server)
    try:
        await _send(writer, request)
        return await _read_to_end(reader)
    finally:
        await _close(writer)


def _content_length(head: bytes) -> int:
    for line in head.split(b"\r\n"):
        name, sep, value = line.partition(b":")
        if sep and name.strip().lower() == b"content-length":
            return int(value.strip())
    return 0


async def _read_response(reader: asyncio.StreamReader) -> bytes:
    head = await asyncio.wait_for(
        reader.readuntil(b"\r\n\r\n"), timeout=TIMEOUT
    )
    body = await asyncio.wait_for(
        reader.readexactly(_content_length(head)), timeout=TIMEOUT
    )
    return head + body


@reset_observable
@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
@pytest.mark.parametrize("max_requests", [None, 1])
async def test_raw_proxy_fault_adapter_preserves_first_drop_reset(
    reset: bool,
    max_requests: int | None,
) -> None:
    forwarder = RawForwarder(
        response_transformer=fault_step_transformer(
            DropConnection(after_bytes=0, reset=reset),
            DropConnection(after_bytes=1, reset=not reset),
        )
    )
    async with (
        AsyncHTTPTestServer() as upstream,
        AsyncHTTPTestServer(raw_forwarder=forwarder) as proxy,
    ):
        proxy.set_keep_alive(max_requests=max_requests)
        request = (
            f"GET {upstream.url}/ HTTP/1.1\r\n"
            f"Host: {upstream.host}:{upstream.port}\r\n\r\n"
        ).encode()
        data, observed_reset = await _exchange_raw(proxy, request)
        closed = await proxy.next_closed_connection(timeout=TIMEOUT)
        exchange = await proxy.next_exchange(timeout=TIMEOUT)

    assert data == b""
    assert observed_reset == reset
    assert closed.reset == reset
    assert exchange.closed is closed
    assert closed.reason == "response_aborted"
    assert closed.phase == "response_body"


@pytest.mark.asyncio
async def test_close_connection_sequence_httpx_fails_then_succeeds() -> None:
    async with AsyncHTTPTestServer() as server, httpx.AsyncClient() as client:
        server.set_response_sequence([
            CloseConnection(),
            HTTPResponse.json({"ok": True}),
        ])

        with pytest.raises(httpx.HTTPError):
            await client.get(server.url)
        closed = await server.next_closed_connection(timeout=TIMEOUT)
        response = await client.get(server.url)

    assert response.json() == {"ok": True}
    assert closed.reason == "close_response"
    assert closed.phase == "response"
    assert not closed.reset
    assert closed.requests_completed == 1
    assert len(server.requests) == 2
    assert server.exchanges[0].closed is closed
    assert server.exchanges[0].response is None
    assert server.exchanges[1].closed is None


@pytest.mark.asyncio
async def test_close_connection_sequence_aiohttp_fails_then_succeeds() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        aiohttp.ClientSession() as session,
    ):
        server.set_response_sequence([
            CloseConnection(),
            HTTPResponse.json({"ok": True}),
        ])

        # POST is not idempotent, so aiohttp surfaces the first failure
        # instead of retrying it transparently.
        with pytest.raises(aiohttp.ClientError):
            await session.post(server.url, data=b"payload")
        closed = await server.next_closed_connection(timeout=TIMEOUT)
        async with session.post(server.url, data=b"payload") as response:
            payload = await response.json()

    assert payload == {"ok": True}
    assert closed.reason == "close_response"
    assert closed.phase == "response"
    assert len(server.requests) == 2
    assert server.exchanges[0].closed is closed
    assert server.exchanges[0].response is None


@pytest.mark.asyncio
async def test_close_connection_aiohttp_retries_idempotent_request() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        aiohttp.ClientSession() as session,
    ):
        server.set_response_sequence([
            CloseConnection(),
            HTTPResponse.json({"ok": True}),
        ])

        async with session.get(server.url) as response:
            payload = await response.json()
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert payload == {"ok": True}
    assert closed.reason == "close_response"
    assert len(server.requests) == 2
    assert server.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_close_connection_drained_raw_client_observes_eof() -> None:
    async with AsyncHTTPTestServer(
        default_response=CloseConnection()
    ) as server:
        data, reset = await _exchange_raw(server, GET_REQUEST)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data == b""
    assert not reset
    assert closed.reason == "close_response"
    assert not closed.reset
    assert closed.bytes_read == len(GET_REQUEST)
    assert closed.bytes_consumed == len(GET_REQUEST)
    assert closed.bytes_written == 0


@pytest.mark.parametrize(
    "body_prefix",
    [b"8\r\nabc", b"3\r\nabc", b"1\r\na\r\n7\r\nbc"],
)
@pytest.mark.asyncio
async def test_close_during_request_stops_at_partial_chunk_budget(
    body_prefix: bytes,
) -> None:
    async with AsyncHTTPTestServer() as server:
        server.use_headers(close_during_request(after_body_bytes=3))
        data, _ = await _exchange_raw(server, CHUNKED_HEAD + body_prefix)
        request = await server.next_request(timeout=TIMEOUT)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data == b""
    assert not request.body_complete
    assert request.body == b"abc"
    assert request.wire_raw_bytes == CHUNKED_HEAD + body_prefix
    assert closed.reason == "request_read"
    assert closed.phase == "request_body"
    assert closed.bytes_consumed == len(request.wire_raw_bytes)


@pytest.mark.asyncio
async def test_close_during_request_spent_budget_closes_at_size_line() -> None:
    upload = CHUNKED_HEAD + b"3\r\nabc\r\n3\r\n"
    async with AsyncHTTPTestServer() as server:
        server.use_headers(close_during_request(after_body_bytes=3))
        data, _ = await _exchange_raw(server, upload)
        request = await server.next_request(timeout=TIMEOUT)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data == b""
    assert not request.body_complete
    assert request.body == b"abc"
    assert request.wire_raw_bytes == CHUNKED_HEAD + b"3\r\nabc\r\n"
    assert closed.reason == "request_read"
    assert closed.phase == "request_body"
    assert closed.bytes_consumed == len(request.wire_raw_bytes)
    assert closed.bytes_read >= closed.bytes_consumed


@reset_observable
@pytest.mark.asyncio
async def test_close_connection_reset_raw_client_observes_reset() -> None:
    async with AsyncHTTPTestServer(
        default_response=CloseConnection(reset=True)
    ) as server:
        data, reset = await _exchange_raw(server, GET_REQUEST)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data == b""
    assert reset
    assert closed.reason == "close_response"
    assert closed.reset


@pytest.mark.asyncio
async def test_close_connection_delay_uses_real_time_by_default() -> None:
    async with AsyncHTTPTestServer(
        default_response=CloseConnection(delay=0.05)
    ) as server:
        started = time.monotonic()
        data, _ = await _exchange_raw(server, GET_REQUEST)
        elapsed = time.monotonic() - started
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data == b""
    assert elapsed >= 0.04
    assert closed.reason == "close_response"
    assert closed.phase == "response"


@pytest.mark.asyncio
async def test_keep_alive_update_while_idle_applies_to_next_request() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        httpx.AsyncClient(timeout=TIMEOUT) as client,
    ):
        first = await client.get(server.url)
        assert first.status_code == 200
        assert first.headers.get("connection") != "close"

        server.set_keep_alive(max_requests=2)
        second = await client.get(server.url)

        assert second.status_code == 200
        assert second.headers.get("connection") == "close"
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert server.requests[0].client == server.requests[1].client
    assert closed.reason == "max_requests"
    assert closed.phase == "after_response"
    assert closed.requests_completed == 2
    assert server.exchanges[0].closed is None
    assert server.exchanges[1].closed is closed


@pytest.mark.asyncio
async def test_close_during_request_stops_content_length_upload() -> None:
    async with AsyncHTTPTestServer() as server:
        server.use_headers(close_during_request(after_body_bytes=12))
        reader, writer = await _connect(server)
        try:
            await _send(writer, UPLOAD_HEAD + b"a" * 10)
            await asyncio.sleep(0.02)
            await _send(writer, b"b" * 10)
            data, _ = await _read_to_end(reader)
        finally:
            await _close(writer)
        request = await server.next_request(timeout=TIMEOUT)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data == b""
    assert request is server.requests[0]
    assert not request.body_complete
    assert request.body == b"a" * 10 + b"b" * 2
    assert request.wire_raw_bytes == UPLOAD_HEAD + b"a" * 10 + b"b" * 2
    assert closed.reason == "request_read"
    assert closed.phase == "request_body"
    assert not closed.reset
    assert closed.bytes_consumed == len(request.wire_raw_bytes)
    assert closed.bytes_read >= closed.bytes_consumed
    exchange = server.exchanges[0]
    assert exchange.request is request
    assert exchange.response is None
    assert exchange.closed is closed


@reset_observable
@pytest.mark.asyncio
async def test_close_during_request_reset_stops_chunked_upload() -> None:
    async with AsyncHTTPTestServer() as server:
        server.use_headers(
            close_during_request(after_body_bytes=7, reset=True)
        )
        reader, writer = await _connect(server)
        try:
            await _send(writer, CHUNKED_HEAD + b"5\r\nhello\r\n")
            await asyncio.sleep(0.02)
            await _send(writer, b"5\r\nworld\r\n0\r\n\r\n")
            data, reset = await _read_to_end(reader)
        finally:
            await _close(writer)
        request = await server.next_request(timeout=TIMEOUT)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data == b""
    assert reset
    assert request is server.requests[0]
    assert not request.body_complete
    assert request.body == b"hellowo"
    assert request.wire_raw_bytes == CHUNKED_HEAD + b"5\r\nhello\r\n5\r\nwo"
    assert closed.reason == "request_read"
    assert closed.phase == "request_body"
    assert closed.reset
    assert server.exchanges[0].response is None
    assert server.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_drop_connection_aborts_response_body() -> None:
    async with AsyncHTTPTestServer() as server:
        server.set_raw_response(RESPONSE_BODY)
        server.set_transmission_strategy(
            FaultyTransmission([DropConnection(after_bytes=15)])
        )
        data, reset = await _exchange_raw(server, GET_REQUEST)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    head, _, body = data.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.1 200 OK\r\n")
    assert body == RESPONSE_BODY[:15]
    assert not reset
    assert closed.reason == "response_aborted"
    assert closed.phase == "response_body"
    assert not closed.reset
    assert closed.bytes_written == len(data)
    exchange = server.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.wire_raw_bytes == data


@reset_observable
@pytest.mark.asyncio
async def test_drop_connection_reset_raw_client_observes_reset() -> None:
    async with AsyncHTTPTestServer() as server:
        server.set_raw_response(RESPONSE_BODY)
        server.set_transmission_strategy(
            FaultyTransmission([DropConnection(after_bytes=15, reset=True)])
        )
        _, reset = await _exchange_raw(server, GET_REQUEST)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert reset
    assert closed.reason == "response_aborted"
    assert closed.reset
    assert server.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_idle_timeout_closes_connection_and_client_reconnects() -> None:
    async with AsyncHTTPTestServer() as server, httpx.AsyncClient() as client:
        server.set_json_response({"ok": True})
        server.set_keep_alive(timeout=0.05)

        first = await client.get(server.url)
        closed = await server.next_closed_connection(timeout=TIMEOUT)
        second = await client.get(server.url)

    assert first.status_code == 200
    assert second.status_code == 200
    assert closed.reason == "idle_timeout"
    assert closed.phase == "idle"
    assert closed.requests_completed == 1
    assert not closed.reset
    assert server.requests[0].client != server.requests[1].client
    assert server.exchanges[0].closed is None


@pytest.mark.asyncio
async def test_max_requests_closes_after_second_response() -> None:
    async with AsyncHTTPTestServer() as server, httpx.AsyncClient() as client:
        server.set_json_response({"ok": True})
        server.set_keep_alive(max_requests=2)

        first = await client.get(server.url)
        second = await client.get(server.url)
        closed = await server.next_closed_connection(timeout=TIMEOUT)
        third = await client.get(server.url)

    assert "connection" not in first.headers
    assert second.headers["connection"] == "close"
    assert third.status_code == 200
    assert closed.reason == "max_requests"
    assert closed.phase == "after_response"
    assert closed.requests_completed == 2
    clients = [request.client for request in server.requests]
    assert clients[0] == clients[1]
    assert clients[1] != clients[2]
    assert server.exchanges[0].closed is None
    assert server.exchanges[1].closed is closed
    assert server.exchanges[2].closed is None


@pytest.mark.asyncio
async def test_client_abort_mid_upload_records_partial_request() -> None:
    head = (
        b"PUT /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 100\r\n\r\n"
    )
    async with AsyncHTTPTestServer() as server:
        _, writer = await _connect(server)
        await _send(writer, head + b"x" * 10)
        await _close(writer)
        request = await server.next_request(timeout=TIMEOUT)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert not request.body_complete
    assert request.body == b"x" * 10
    assert request.wire_raw_bytes == head + b"x" * 10
    assert closed.reason == "client"
    assert closed.phase == "request_body"
    assert not closed.reset
    assert closed.requests_completed == 0
    exchange = server.exchanges[0]
    assert exchange.request is request
    assert exchange.response is None
    assert exchange.closed is closed


@pytest.mark.asyncio
async def test_header_phase_continue_then_reject_records_both_sends() -> None:
    async def reject_after_continue(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(HTTPResponse(status=100))
        await ctx.send(HTTPResponse.text("Forbidden", status=403))
        return CloseDuringRequest()

    head = (
        b"PUT /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 10\r\n"
        b"Expect: 100-continue\r\n\r\n"
    )
    async with AsyncHTTPTestServer() as server:
        server.use_headers(reject_after_continue)
        data, _ = await _exchange_raw(server, head)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data.startswith(
        b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 403 Forbidden\r\n"
    )
    assert data.endswith(b"\r\n\r\nForbidden")
    assert closed.reason == "request_read"
    assert closed.phase == "request_body"
    assert closed.bytes_written == len(data)
    exchange = server.exchanges[0]
    assert exchange.closed is closed
    assert not exchange.request.body_complete
    assert exchange.request.body == b""
    assert exchange.response is not None
    assert exchange.response.status == 403
    assert [r.status for r in exchange.interim_responses] == [100]


@pytest.mark.asyncio
async def test_header_phase_final_send_then_true_skips_responder() -> None:
    handled: list[str] = []

    def handler(ctx: ResponderContext) -> HTTPResponse:
        handled.append(ctx.request.target)
        return HTTPResponse.text("handled")

    async def accept_uploads(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        if ctx.headers.method != "POST":
            return await call_next()
        await ctx.send(HTTPResponse.text("accepted", status=202))
        return True

    async with AsyncHTTPTestServer(handler=handler) as server:
        server.use_headers(accept_uploads)
        reader, writer = await _connect(server)
        try:
            await _send(
                writer,
                b"POST /upload HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Content-Length: 5\r\n\r\nhello",
            )
            first = await _read_response(reader)
            await _send(
                writer, b"GET /next HTTP/1.1\r\nHost: localhost\r\n\r\n"
            )
            second = await _read_response(reader)
        finally:
            await _close(writer)

    assert first.startswith(b"HTTP/1.1 202 Accepted\r\n")
    assert first.endswith(b"accepted")
    assert second.endswith(b"handled")
    assert handled == ["/next"]
    upload, follow_up = server.exchanges
    assert upload.request.body == b"hello"
    assert upload.request.body_complete
    assert upload.response is not None
    assert upload.response.status == 202
    assert upload.closed is None
    assert follow_up.response is not None
    assert follow_up.response.status == 200


@pytest.mark.asyncio
async def test_header_phase_second_final_send_raises_runtime_error() -> None:
    errors: list[RuntimeError] = []

    async def send_twice(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(HTTPResponse.text("first"))
        try:
            await ctx.send(HTTPResponse.text("second"))
        except RuntimeError as exc:
            errors.append(exc)
        return True

    async with AsyncHTTPTestServer() as server:
        server.use_headers(send_twice)
        reader, writer = await _connect(server)
        try:
            await _send(writer, GET_REQUEST)
            response = await _read_response(reader)
        finally:
            await _close(writer)

    assert len(errors) == 1
    assert response.endswith(b"first")
    exchange = server.exchanges[0]
    assert exchange.response is not None
    assert exchange.response.body == b"first"
    assert exchange.closed is None


@pytest.mark.asyncio
async def test_header_phase_unhandled_second_send_closes_with_error() -> None:
    async def send_twice(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(HTTPResponse.text("first"))
        await ctx.send(HTTPResponse.text("second"))
        return True

    async with AsyncHTTPTestServer() as server:
        server.use_headers(send_twice)
        data, _ = await _exchange_raw(server, GET_REQUEST)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data.startswith(b"HTTP/1.1 200 OK\r\n")
    assert data.endswith(b"first")
    assert closed.reason == "error"
    assert closed.phase == "response"
    assert closed.bytes_written == len(data)
    # The headers completed, so the recording matrix records the request
    # and an exchange owning the error close and the one final send.
    assert len(server.requests) == 1
    assert server.requests[0].body_complete
    exchange = server.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.body == b"first"


@pytest.mark.parametrize(
    ("timeout", "max_requests", "hint"),
    [
        (5, 3, "timeout=5, max=3"),
        (0.2, 3, "max=3"),
        (0.2, None, None),
    ],
)
@pytest.mark.asyncio
async def test_keep_alive_advertise_header(
    timeout: float, max_requests: int | None, hint: str | None
) -> None:
    async with AsyncHTTPTestServer() as server, httpx.AsyncClient() as client:
        server.set_json_response({"ok": True})
        server.set_keep_alive(
            timeout=timeout, max_requests=max_requests, advertise=True
        )
        response = await client.get(server.url)

    assert response.status_code == 200
    if hint is None:
        assert "keep-alive" not in response.headers
        assert "connection" not in response.headers
    else:
        assert response.headers["keep-alive"] == hint
        assert response.headers["connection"] == "keep-alive"


@pytest.mark.asyncio
async def test_zero_timeout_counts_pipelined_request_as_read_ahead() -> None:
    first = b"GET /one HTTP/1.1\r\nHost: localhost\r\n\r\n"
    second = b"GET /two HTTP/1.1\r\nHost: localhost\r\n\r\n"
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"ok": True})
        server.set_keep_alive(timeout=0.0)
        data, _ = await _exchange_raw(server, first + second)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert data.count(b"HTTP/1.1 200 OK\r\n") == 1
    assert b"Connection: close" not in data
    assert closed.reason == "idle_timeout"
    assert closed.phase == "after_response"
    assert closed.requests_completed == 1
    assert closed.bytes_consumed == len(first)
    assert closed.bytes_read - closed.bytes_consumed == len(second)
    assert closed.bytes_written == len(data)
    assert [r.target for r in server.requests] == ["/one"]
    assert server.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_counters_available_without_connection_byte_retention() -> None:
    request = b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    async with AsyncHTTPTestServer(max_connection_bytes=0) as server:
        server.set_json_response({"ok": True})
        data, _ = await _exchange_raw(server, request)
        closed = await server.next_closed_connection(timeout=TIMEOUT)

    assert closed.reason == "connection_close"
    assert closed.phase == "after_response"
    assert closed.bytes_read == len(request)
    assert closed.bytes_consumed == len(request)
    assert closed.bytes_written == len(data)
    assert closed.client is not None
    assert server.get_connection_bytes_received(closed.client) is None
    assert server.get_connection_bytes_sent(closed.client) is None
    assert server.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_aclose_records_shutdown_for_idle_raw_client() -> None:
    server = AsyncHTTPTestServer()
    await server.start()
    reader, writer = await _connect(server)
    try:
        await _send(writer, GET_REQUEST)
        await _read_response(reader)
        await asyncio.wait_for(server.aclose(), timeout=TIMEOUT)
        data, _ = await _read_to_end(reader)
    finally:
        await _close(writer)
        await server.aclose()

    assert data == b""
    closed = server.closed_connections[0]
    assert closed.reason == "shutdown"
    assert closed.phase == "idle"
    assert closed.requests_completed == 1
    assert server.exchanges[0].closed is None
