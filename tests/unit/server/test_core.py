from __future__ import annotations

import asyncio
import socket
from typing import Any
from unittest.mock import Mock, create_autospec
from urllib.parse import urlsplit

import pytest

from localstub.forward import RawForwarder
from localstub.http.client import HTTPClient
from localstub.http.clients.asyncio import AsyncioClient
from localstub.http.exchange import ConnectionClosed
from localstub.http.headers import Headers
from localstub.http.request import HTTPRequestHeaders, HTTPRequestReader
from localstub.middleware import (
    CloseConnection,
    HeaderContext,
    HeaderDecision,
    HeaderNext,
    ResponderContext,
    ResponseSpec,
    SenderContext,
    SenderNext,
    SendResult,
)
from localstub.recording import TrafficRecorder
from localstub.server import (
    AsyncHTTPTestServer,
    HTTPResponse,
    RecordedHTTPRequest,
    SendResponse,
)
from localstub.throttle import ThrottleDecision

CLIENT = ("127.0.0.1", 4321)
GET = b"GET /one HTTP/1.1\r\nHost: localhost\r\n\r\n"
GET_TWO = b"GET /two HTTP/1.1\r\nHost: localhost\r\n\r\n"
POST_HEAD = (
    b"POST /up HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\n\r\n"
)
PROXY_GET = (
    b"GET http://upstream.test/x HTTP/1.1\r\nHost: upstream.test\r\n\r\n"
)


class _RecordingSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def _fake_writer(peer: tuple[str, int] | None = CLIENT) -> Mock:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    writer.is_closing.return_value = False

    def get_extra_info(name: str, default: Any = None) -> Any:
        return peer if name == "peername" else default

    def close() -> None:
        writer.is_closing.return_value = True

    writer.get_extra_info.side_effect = get_extra_info
    writer.close.side_effect = close
    writer.transport = create_autospec(asyncio.Transport, instance=True)
    writer.transport.abort.side_effect = close
    return writer


def _reader(data: bytes = b"", *, eof: bool = True) -> asyncio.StreamReader:
    reader = asyncio.StreamReader()
    if data:
        reader.feed_data(data)
    if eof:
        reader.feed_eof()
    return reader


def _written(writer: Mock) -> bytes:
    return b"".join(call.args[0] for call in writer.write.call_args_list)


def _absolute_get(url: str) -> bytes:
    host = urlsplit(url).netloc
    return f"GET {url} HTTP/1.1\r\nHost: {host}\r\n\r\n".encode("ascii")


async def _converse(
    server: AsyncHTTPTestServer,
    data: bytes,
    *,
    eof: bool = True,
    writer: Mock | None = None,
) -> ConnectionClosed:
    writer = _fake_writer() if writer is None else writer
    await server.handle_http_connection(_reader(data, eof=eof), writer)
    closed = server.last_closed_connection
    assert closed is not None
    return closed


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _expect_eof(sock: socket.socket) -> None:
    loop = asyncio.get_running_loop()
    try:
        data = await asyncio.wait_for(loop.sock_recv(sock, 1), timeout=1.0)
    except ConnectionResetError:
        return
    assert data == b""


def _listening(server: AsyncHTTPTestServer) -> tuple[str, int]:
    assert server.host is not None
    assert server.port is not None
    return server.host, server.port


def test_server_url_raises_when_not_started() -> None:
    server = AsyncHTTPTestServer()
    with pytest.raises(RuntimeError, match="Server not started yet"):
        _ = server.url


def test_forward_proxy_with_injected_client_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not both"):
        AsyncHTTPTestServer(
            forward_proxy=True, upstream_client=AsyncioClient()
        )


def test_server_handler_getter() -> None:
    def handler(ctx: ResponderContext) -> HTTPResponse:
        _ = ctx
        return HTTPResponse.json({"test": "value"})

    server = AsyncHTTPTestServer(handler=handler)
    assert server.handler is handler


def test_server_accepts_recorder_in_legacy_positional_slot() -> None:
    recorder = TrafficRecorder(None)

    server = AsyncHTTPTestServer(
        "127.0.0.1",
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        recorder,
    )

    assert server.requests == []


def test_zero_recording_buffer_size_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        AsyncHTTPTestServer(recording_buffer_size=0)


def test_negative_recording_buffer_size_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        AsyncHTTPTestServer(recording_buffer_size=-1)


def test_negative_max_connection_bytes_raises_value_error() -> None:
    with pytest.raises(ValueError, match="must be non-negative or None"):
        AsyncHTTPTestServer(max_connection_bytes=-1)


def test_default_response_getter_returns_close_connection() -> None:
    close = CloseConnection()
    server = AsyncHTTPTestServer()

    server.default_response = close

    assert server.default_response is close


@pytest.mark.asyncio
async def test_close_connection_default_response_closes_each_request() -> None:
    server = AsyncHTTPTestServer(default_response=CloseConnection())

    closed = await _converse(server, GET)

    assert closed.reason == "close_response"
    assert server.last_exchange is not None
    assert server.last_exchange.closed is closed


@pytest.mark.asyncio
async def test_constructor_max_requests_per_connection_seeds_policy() -> None:
    server = AsyncHTTPTestServer(max_requests_per_connection=1)
    writer = _fake_writer()

    closed = await _converse(server, GET + GET_TWO, writer=writer)

    assert closed.reason == "max_requests"
    assert len(server.requests) == 1
    assert b"Connection: close\r\n" in _written(writer)


@pytest.mark.asyncio
async def test_constructor_keep_alive_timeout_seeds_policy_via_sleep() -> None:
    sleep = _RecordingSleep()
    server = AsyncHTTPTestServer(keep_alive_timeout=0.2, sleep=sleep)

    closed = await _converse(server, GET, eof=False)

    assert closed.reason == "idle_timeout"
    assert sleep.calls == [0.2]


def test_set_keep_alive_rejects_negative_timeout() -> None:
    server = AsyncHTTPTestServer()

    with pytest.raises(ValueError, match="timeout"):
        server.set_keep_alive(timeout=-0.1)


def test_set_keep_alive_rejects_max_requests_below_one() -> None:
    server = AsyncHTTPTestServer()

    with pytest.raises(ValueError, match="max_requests"):
        server.set_keep_alive(max_requests=0)


@pytest.mark.asyncio
async def test_set_keep_alive_replaces_every_earlier_setting() -> None:
    server = AsyncHTTPTestServer()
    server.set_keep_alive(max_requests=1, advertise=True)
    writer = _fake_writer()

    server.set_keep_alive(timeout=5)
    closed = await _converse(server, GET + GET_TWO, writer=writer)

    assert closed.reason == "client"
    assert len(server.requests) == 2
    assert b"Keep-Alive:" not in _written(writer)


@pytest.mark.asyncio
async def test_use_headers_runs_registered_header_middleware() -> None:
    async def refuse(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = (ctx, call_next)
        return False

    server = AsyncHTTPTestServer()
    server.use_headers(refuse)

    closed = await _converse(server, POST_HEAD + b"0123456789")

    assert closed.reason == "request_read"
    assert server.requests[0].body == b""


@pytest.mark.asyncio
async def test_set_request_headers_handler_false_closes_during_request() -> (
    None
):
    def refuse(headers: HTTPRequestHeaders, send: SendResponse) -> bool:
        _ = (headers, send)
        return False

    server = AsyncHTTPTestServer()
    server.set_request_headers_handler(refuse)

    closed = await _converse(server, POST_HEAD + b"0123456789")

    assert closed.reason == "request_read"
    assert closed.phase == "request_body"
    assert not server.requests[0].body_complete


@pytest.mark.asyncio
async def test_set_throttle_without_key_throttles_every_request_together() -> (
    None
):
    server = AsyncHTTPTestServer()
    server.set_throttle(rate_per_second=0.01, burst=1)

    await _converse(server, GET + GET_TWO)

    assert [r.status for r in server.responses] == [200, 429]


@pytest.mark.asyncio
async def test_set_throttle_callable_response_builds_throttled_reply() -> None:
    def reply(
        request: RecordedHTTPRequest, decision: ThrottleDecision
    ) -> HTTPResponse:
        _ = decision
        return HTTPResponse.text(f"slow down {request.target}", status=503)

    server = AsyncHTTPTestServer()
    server.set_throttle(rate_per_second=0.01, burst=1, response=reply)

    await _converse(server, GET + GET_TWO)

    assert server.responses[1].status == 503
    assert server.responses[1].body == b"slow down /two"


@pytest.mark.asyncio
async def test_start_twice_keeps_the_first_listener() -> None:
    server = AsyncHTTPTestServer()
    await server.start()
    try:
        url = server.url

        await server.start()

        assert server.url == url
    finally:
        await server.aclose()


@pytest.mark.asyncio
async def test_aclose_without_listener_shuts_down_handed_off_connection() -> (
    None
):
    server = AsyncHTTPTestServer()
    task = asyncio.create_task(
        server.handle_http_connection(_reader(eof=False), _fake_writer())
    )
    await _settle()

    await server.aclose()
    await asyncio.wait_for(task, timeout=1.0)

    closed = server.last_closed_connection
    assert closed is not None
    assert closed.reason == "shutdown"
    assert closed.phase == "idle"


@pytest.mark.asyncio
async def test_close_http_connection_shuts_down_the_tracked_connection() -> (
    None
):
    server = AsyncHTTPTestServer()
    writer = _fake_writer()
    task = asyncio.create_task(
        server.handle_http_connection(_reader(eof=False), writer)
    )
    await _settle()

    server.close_http_connection(writer)
    await asyncio.wait_for(task, timeout=1.0)

    assert [c.reason for c in server.closed_connections] == ["shutdown"]
    writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_close_http_connection_after_finish_records_nothing_more() -> (
    None
):
    server = AsyncHTTPTestServer()
    writer = _fake_writer()
    closed = await _converse(server, GET, writer=writer)

    server.close_http_connection(writer)
    server.close_http_connection(writer)

    assert server.closed_connections == [closed]


def test_close_http_connection_ignores_untracked_writer() -> None:
    server = AsyncHTTPTestServer()

    server.close_http_connection(_fake_writer())

    assert server.closed_connections == []


@pytest.mark.asyncio
async def test_connection_handed_off_while_closing_records_shutdown_idle() -> (
    None
):
    server = AsyncHTTPTestServer()
    await server.start()
    closing = asyncio.create_task(server.aclose())
    await asyncio.sleep(0)
    writer = _fake_writer()

    await server.handle_http_connection(_reader(GET, eof=False), writer)
    await asyncio.wait_for(closing, timeout=1.0)

    closed = server.last_closed_connection
    assert closed is not None
    assert closed.reason == "shutdown"
    assert closed.phase == "idle"
    assert closed.bytes_read == 0
    assert server.requests == []
    writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_handler_calling_aclose_on_own_connection_finishes_request() -> (
    None
):
    server = AsyncHTTPTestServer()

    async def handler(ctx: ResponderContext) -> HTTPResponse:
        _ = ctx
        await server.aclose()
        return HTTPResponse.text("done")

    server.handler = handler
    writer = _fake_writer()

    closed = await asyncio.wait_for(
        _converse(server, GET + GET_TWO, writer=writer), timeout=1.0
    )

    assert _written(writer).endswith(b"done")
    assert len(server.requests) == 1
    assert closed.reason == "shutdown"
    assert closed.phase == "response"
    assert server.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_handler_calling_aclose_on_started_server_finishes_request() -> (
    None
):
    server = AsyncHTTPTestServer()

    async def handler(ctx: ResponderContext) -> HTTPResponse:
        _ = ctx
        await server.aclose()
        return HTTPResponse.text("done")

    server.handler = handler
    await server.start()
    reader, writer = await asyncio.open_connection(*_listening(server))
    try:
        writer.write(GET)
        response = await asyncio.wait_for(reader.read(), timeout=1.0)
    finally:
        writer.close()
    closed = await server.next_closed_connection(timeout=1.0)

    assert response.endswith(b"done")
    assert closed.reason == "shutdown"
    assert closed.phase == "response"
    assert server.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_handler_calling_aclose_on_started_server_closes_others() -> (
    None
):
    server = AsyncHTTPTestServer()

    async def handler(ctx: ResponderContext) -> HTTPResponse:
        _ = ctx
        await server.aclose()
        return HTTPResponse.text("done")

    server.handler = handler
    await server.start()
    idle = socket.create_connection(_listening(server))
    idle.setblocking(False)
    reader, writer = await asyncio.open_connection(*_listening(server))
    try:
        await _settle()
        writer.write(GET)
        response = await asyncio.wait_for(reader.read(), timeout=1.0)
        await _expect_eof(idle)
    finally:
        writer.close()
        idle.close()
    first = await server.next_closed_connection(timeout=1.0)
    second = await server.next_closed_connection(timeout=1.0)

    assert response.endswith(b"done")
    assert (first.phase, second.phase) == ("idle", "response")
    assert {first.reason, second.reason} == {"shutdown"}


# A blocking connect completes the handshake without yielding, so the
# number of turns taken afterwards selects how far asyncio's accept
# pipeline has progressed when aclose() starts: two or three turns
# leave the socket accepted but not yet handed to the server, which
# then sees it during shutdown; more turns register it as a live
# connection that shutdown cancels.  Both must record the same event.
@pytest.mark.parametrize("accept_turns", [2, 3, 4, 8])
@pytest.mark.asyncio
async def test_aclose_shuts_down_accepted_idle_connection(
    accept_turns: int,
) -> None:
    server = AsyncHTTPTestServer()
    await server.start()
    sock = socket.create_connection(_listening(server))
    sock.setblocking(False)
    try:
        for _ in range(accept_turns):
            await asyncio.sleep(0)

        await asyncio.wait_for(server.aclose(), timeout=1.0)

        await _expect_eof(sock)
    finally:
        sock.close()
    closed = server.last_closed_connection
    assert closed is not None
    assert closed.reason == "shutdown"
    assert closed.phase == "idle"


@pytest.mark.asyncio
async def test_cancelled_aclose_finishes_shutdown_then_reraises() -> None:
    server = AsyncHTTPTestServer()
    await server.start()
    host, port = _listening(server)
    closing = asyncio.create_task(server.aclose())
    await asyncio.sleep(0)

    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    with pytest.raises(OSError):
        await asyncio.open_connection(host, port)
    await server.start()
    await server.aclose()


@pytest.mark.asyncio
async def test_concurrent_aclose_calls_both_complete() -> None:
    server = AsyncHTTPTestServer()
    await server.start()
    host, port = _listening(server)

    await asyncio.gather(server.aclose(), server.aclose())

    with pytest.raises(OSError):
        await asyncio.open_connection(host, port)


@pytest.mark.asyncio
async def test_injected_upstream_client_serves_absolute_form_requests() -> (
    None
):
    client = create_autospec(HTTPClient, instance=True)
    client.send.return_value = HTTPResponse.text("from upstream")
    server = AsyncHTTPTestServer(upstream_client=client)

    await _converse(server, PROXY_GET)

    client.send.assert_awaited_once()
    assert server.responses[0].body == b"from upstream"


@pytest.mark.asyncio
async def test_aclose_never_closes_an_injected_upstream_client() -> None:
    client = create_autospec(HTTPClient, instance=True)
    server = AsyncHTTPTestServer(upstream_client=client)

    await server.aclose()

    client.aclose.assert_not_awaited()


@pytest.mark.asyncio
async def test_raw_forwarder_relays_absolute_form_requests() -> None:
    forwarder = create_autospec(RawForwarder, instance=True)
    forwarder.forward_and_relay.return_value = None
    server = AsyncHTTPTestServer(raw_forwarder=forwarder)

    await _converse(server, PROXY_GET)

    forwarder.forward_and_relay.assert_awaited_once()
    assert server.responses[0].status == 502


@pytest.mark.asyncio
async def test_forward_proxy_recreates_owned_client_after_aclose() -> None:
    async with AsyncHTTPTestServer() as upstream:
        upstream.set_text_response("from upstream")
        proxy = AsyncHTTPTestServer(forward_proxy=True)
        await proxy.aclose()
        await proxy.start()
        try:
            await _converse(proxy, _absolute_get(upstream.url + "x"))
        finally:
            await proxy.aclose()

        assert [r.target for r in upstream.requests] == ["/x"]
        assert proxy.responses[0].body == b"from upstream"


@pytest.mark.asyncio
async def test_concurrent_aclose_closes_pooled_upstream_connection() -> None:
    async with AsyncHTTPTestServer() as upstream:
        proxy = AsyncHTTPTestServer(forward_proxy=True)
        await proxy.start()
        await _converse(proxy, _absolute_get(upstream.url))

        await asyncio.gather(proxy.aclose(), proxy.aclose())

        closed = await upstream.next_closed_connection(timeout=1.0)
        assert closed.reason == "client"
        assert closed.requests_completed == 1


@pytest.mark.asyncio
async def test_retained_bytes_accumulate_across_a_clients_connections() -> (
    None
):
    server = AsyncHTTPTestServer()

    await _converse(server, GET)
    await _converse(server, GET_TWO)

    assert server.get_connection_bytes_received(CLIENT) == GET + GET_TWO
    assert len(server.closed_connections) == 2


@pytest.mark.asyncio
async def test_max_connection_bytes_zero_still_counts_bytes() -> None:
    server = AsyncHTTPTestServer(max_connection_bytes=0)
    writer = _fake_writer()

    closed = await _converse(server, GET, writer=writer)

    assert closed.bytes_read == len(GET)
    assert closed.bytes_consumed == len(GET)
    assert closed.bytes_written == len(_written(writer))
    assert server.get_connection_bytes_received(CLIENT) is None
    assert server.get_connection_bytes_sent(CLIENT) is None


@pytest.mark.asyncio
async def test_same_writer_handed_off_twice_keeps_second_tracked() -> None:
    server = AsyncHTTPTestServer()
    writer = _fake_writer()
    reader_one = _reader(eof=False)
    first = asyncio.create_task(
        server.handle_http_connection(reader_one, writer)
    )
    second = asyncio.create_task(
        server.handle_http_connection(_reader(eof=False), writer)
    )
    await _settle()

    reader_one.feed_eof()
    await asyncio.wait_for(first, timeout=1.0)
    server.close_http_connection(writer)
    await asyncio.wait_for(second, timeout=1.0)

    assert [c.reason for c in server.closed_connections] == [
        "client",
        "shutdown",
    ]


@pytest.mark.asyncio
async def test_handed_off_reader_read_ahead_is_served_and_recorded() -> None:
    server = AsyncHTTPTestServer()
    writer = _fake_writer()
    reader = _reader(GET + GET_TWO)
    first = await HTTPRequestReader().read_request(reader)
    assert first is not None
    assert first.target == "/one"

    await server.handle_http_connection(reader, writer)
    closed = server.last_closed_connection

    assert [request.target for request in server.requests] == ["/two"]
    assert closed is not None
    assert closed.reason == "client"
    assert closed.requests_completed == 1
    assert closed.bytes_read == len(GET_TWO)
    assert closed.bytes_consumed == len(GET_TWO)


@pytest.mark.asyncio
async def test_closed_connection_accessors_expose_the_recorded_event() -> None:
    server = AsyncHTTPTestServer()

    closed = await _converse(server, GET)

    assert server.closed_connections == [closed]
    assert server.last_closed_connection is closed
    assert await server.next_closed_connection(timeout=1.0) is closed
    assert server.dropped_closed_connections == 0


@pytest.mark.asyncio
async def test_dropped_closed_connections_counts_unread_evictions() -> None:
    server = AsyncHTTPTestServer(recording_buffer_size=1)

    await _converse(server, GET)
    await _converse(server, GET_TWO)

    assert server.dropped_closed_connections == 1
    assert len(server.closed_connections) == 1


@pytest.mark.asyncio
async def test_next_exchange_returns_the_completed_exchange() -> None:
    server = AsyncHTTPTestServer()

    await _converse(server, GET)
    exchange = await server.next_exchange(timeout=1.0)

    assert exchange.request.target == "/one"
    assert exchange.response is not None
    assert exchange.closed is None


@pytest.mark.asyncio
async def test_repeated_aclose_without_listener_is_idempotent() -> None:
    proxy = AsyncHTTPTestServer(forward_proxy=True)

    await proxy.aclose()
    await proxy.aclose()

    assert proxy.closed_connections == []


@pytest.mark.asyncio
async def test_static_response_still_runs_sender_middleware() -> None:
    response = HTTPResponse.text("default")
    server = AsyncHTTPTestServer(default_response=response)
    seen: list[str] = []

    async def sender(
        ctx: SenderContext, spec: ResponseSpec, call_next: SenderNext
    ) -> SendResult:
        assert spec is response
        assert ctx.connection.client == CLIENT
        seen.append(ctx.request.target)
        return await call_next(HTTPResponse.text("wrapped", status=202))

    server.use_sender(sender)
    await _converse(server, GET)

    assert seen == ["/one"]
    assert server.last_response is not None
    assert server.last_response.status == 202
    assert server.last_response.body == b"wrapped"


@pytest.mark.asyncio
async def test_cached_head_tracks_response_mutations_and_live_policy() -> None:
    response = HTTPResponse(body=b"x")
    server = AsyncHTTPTestServer(default_response=response)
    reader = _reader(eof=False)
    writer = _fake_writer()
    task = asyncio.create_task(server.handle_http_connection(reader, writer))
    cases = [
        (200, "a", b"x", b"X-Test: a\r\nContent-Length: 1\r\n\r\nx"),
        (
            200,
            "a",
            b"longer",
            b"X-Test: a\r\nContent-Length: 6\r\n\r\nlonger",
        ),
        (
            200,
            "b",
            b"longer",
            b"X-Test: b\r\nContent-Length: 6\r\n\r\nlonger",
        ),
        (204, "b", b"longer", b"X-Test: b\r\n\r\n"),
        (
            200,
            "b",
            b"longer",
            (
                b"X-Test: b\r\nContent-Length: 6\r\n"
                b"Keep-Alive: timeout=5\r\n"
                b"Connection: keep-alive\r\n\r\nlonger"
            ),
        ),
        (
            200,
            "b",
            b"longer",
            (
                b"X-Test: b\r\nContent-Length: 6\r\n"
                b"Connection: close\r\n\r\nlonger"
            ),
        ),
    ]
    try:
        for index, (status, tag, body, expected) in enumerate(cases):
            response.status = status
            response.headers = Headers.from_items([("X-Test", tag)])
            response.body = body
            if index == 4:
                server.set_keep_alive(timeout=5, advertise=True)
            elif index == 5:
                server.set_keep_alive(max_requests=6)
            reader.feed_data(GET)
            exchange = await server.next_exchange(timeout=1.0)
            assert exchange.response is not None
            line = (
                b"HTTP/1.1 204 No Content\r\n"
                if status == 204
                else b"HTTP/1.1 200 OK\r\n"
            )
            assert exchange.response.wire_raw_bytes == line + expected
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        reader.feed_eof()
        await server.aclose()
        await asyncio.wait_for(task, timeout=1.0)

    assert server.last_closed_connection is not None
    assert server.last_closed_connection.reason == "max_requests"
    assert server.last_closed_connection.requests_completed == 6
    assert server.responses[0].body == b"x"
