from __future__ import annotations

import asyncio
import dataclasses
import logging
import socket
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.message import Message
from typing import Any
from unittest.mock import AsyncMock, Mock, create_autospec

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from localstub.forward import ForwardResult, RawForwarder, RelayedResponse
from localstub.http.exchange import ConnectionClosed
from localstub.http.headers import Headers
from localstub.http.stream import ByteStream, unread_data
from localstub.http.utils import maybe_await
from localstub.middleware import (
    CloseConnection,
    CloseDuringRequest,
    ForwardProxyResponse,
    HeaderContext,
    HeaderDecision,
    HeaderMiddleware,
    HeaderNext,
    ResponderContext,
    ResponseSpec,
    SenderContext,
    SenderMiddleware,
    SenderNext,
    SendResult,
    ServerServices,
    TimestampProvider,
    compose_headers,
    compose_sender,
)
from localstub.recording import BoundedByteBuffer, TrafficRecorder
from localstub.server import (
    AbortTransmission,
    ConnectionState,
    CountingStreamReader,
    DropConnection,
    FaultyTransmission,
    HTTPConnection,
    HTTPResponse,
    ImmediateTransmission,
    KeepAlivePolicy,
    RecordingStreamWriter,
    RequestPipeline,
    ThrottledTransmission,
    TransmissionStrategy,
    Writer,
    pack_linger_option,
)
from localstub.server.connection import (
    CaptureContext,
    ResponderApp,
    SenderApp,
)
from localstub.server.transmission import Delay, FaultStep

CLIENT = ("127.0.0.1", 4321)
TIMESTAMP = datetime(2026, 9, 7, tzinfo=UTC)
GET = b"GET /one HTTP/1.1\r\nHost: localhost\r\n\r\n"
GET_TWO = b"GET /two HTTP/1.1\r\nHost: localhost\r\n\r\n"
POST_HEAD = (
    b"POST /up HTTP/1.1\r\nHost: localhost\r\nContent-Length: 10\r\n\r\n"
)
CHUNKED_HEAD = (
    b"POST /up HTTP/1.1\r\nHost: localhost\r\n"
    b"Transfer-Encoding: chunked\r\n\r\n"
)
CHUNKED_BODY = b"5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n"
UPGRADE_HEAD = (
    b"POST /chat HTTP/1.1\r\nHost: localhost\r\n"
    b"Connection: Upgrade\r\nUpgrade: custom\r\nContent-Length: 0\r\n\r\n"
)
HEAD_ONE = b"HEAD /one HTTP/1.1\r\nHost: localhost\r\n\r\n"
GZIP_HEAD = (
    b"POST /up HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: gzip\r\n\r\n"
)
DEFAULT_RESPONSE = HTTPResponse.json({"ok": True})
JSON_RESPONSE = (
    b"HTTP/1.1 200 OK\r\n"
    b"Content-Type: application/json\r\n"
    b"Content-Length: 12\r\n"
    b"\r\n"
    b'{"ok": true}'
)

type Responder = Callable[
    [ResponderContext], Awaitable[ResponseSpec] | ResponseSpec
]


class _FixedTimestampProvider:
    def now(self) -> datetime:
        return TIMESTAMP


class _RaisingTimestampProvider:
    def now(self) -> datetime:
        raise RuntimeError("no timestamps left")


class _ImmediateSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


class _HeldSleep:
    def __init__(self) -> None:
        self.calls: list[float] = []
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.entered.set()
        await self.release.wait()


class _ScriptedReader:
    """A stream whose reads return scripted chunks or raise."""

    def __init__(self, script: list[bytes | Exception]) -> None:
        self._script = script

    async def read(self, n: int = -1) -> bytes:
        if not self._script:
            return b""
        item = self._script[0]
        if isinstance(item, Exception):
            self._script.pop(0)
            raise item
        data, rest = (item, b"") if n < 0 else (item[:n], item[n:])
        if rest:
            self._script[0] = rest
        else:
            self._script.pop(0)
        return data


class _RaisingTransmission(TransmissionStrategy):
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def write_body(
        self, writer: Writer, body: bytes
    ) -> AbortTransmission | None:
        _ = (writer, body)
        raise self._error


def _fake_writer(sock: socket.socket | None = None) -> Mock:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    writer.is_closing.return_value = False
    writer.get_extra_info.return_value = sock

    def close() -> None:
        writer.is_closing.return_value = True

    writer.close.side_effect = close
    writer.transport = create_autospec(asyncio.Transport, instance=True)
    writer.transport.abort.side_effect = close
    return writer


def _held_close_writer() -> tuple[Mock, asyncio.Event]:
    """A writer whose close finishes only once the returned event is set."""
    release = asyncio.Event()

    async def wait_closed() -> None:
        await release.wait()

    writer = _fake_writer()
    writer.wait_closed.side_effect = wait_closed
    return writer, release


@dataclass
class Harness:
    reader: ByteStream
    writer: Mock
    state: ConnectionState
    recorder: TrafficRecorder
    connection: HTTPConnection
    responded: list[str] = field(default_factory=list[str])

    @property
    def written(self) -> bytes:
        return b"".join(
            call.args[0] for call in self.writer.write.call_args_list
        )

    @property
    def closed(self) -> ConnectionClosed:
        assert len(self.recorder.closed_connections) == 1
        return self.recorder.closed_connections[0]

    def feed(self, data: bytes = b"", *, eof: bool = False) -> None:
        reader = self.reader
        assert isinstance(reader, asyncio.StreamReader)
        if data:
            reader.feed_data(data)
        if eof:
            reader.feed_eof()

    async def run(self) -> ConnectionClosed:
        await self.connection.run()
        return self.closed


def _harness(
    *,
    response: ResponseSpec | Responder = DEFAULT_RESPONSE,
    header_middlewares: list[HeaderMiddleware] | None = None,
    sender_middlewares: list[SenderMiddleware] | None = None,
    transmission: TransmissionStrategy | None = None,
    keep_alive: KeepAlivePolicy | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    writer: Mock | None = None,
    reader: ByteStream | None = None,
    client: tuple[str, int] | None = CLIENT,
    received: BoundedByteBuffer | None = None,
    sent: BoundedByteBuffer | None = None,
    timestamp_provider: TimestampProvider | None = None,
) -> Harness:
    responded: list[str] = []
    stream_reader = asyncio.StreamReader() if reader is None else reader
    stream_writer = _fake_writer() if writer is None else writer
    state = ConnectionState(client=client, received=received, sent=sent)
    recorder = TrafficRecorder(None)
    policy = keep_alive or KeepAlivePolicy()
    strategy = transmission or ImmediateTransmission()

    def responder(capture: CaptureContext | None) -> ResponderApp:
        async def app(ctx: ResponderContext) -> ResponseSpec:
            responded.append(ctx.request.target)
            if capture is not None:
                capture(ctx)
            if callable(response):
                return await maybe_await(response(ctx))
            return response

        return app

    async def header_terminal(_: HeaderContext) -> HeaderDecision:
        return True

    header_app = (
        compose_headers(header_middlewares, header_terminal)
        if header_middlewares
        else None
    )

    def sender(terminal: SenderApp) -> SenderApp | None:
        if not sender_middlewares:
            return None
        return compose_sender(sender_middlewares, terminal)

    pipeline = RequestPipeline(
        header=lambda: header_app,
        responder=responder,
        sender=sender,
        transmission=lambda: strategy,
        keep_alive=lambda: policy,
    )
    connection = HTTPConnection(
        reader=CountingStreamReader(stream_reader, state),
        writer=RecordingStreamWriter(stream_writer, sent, state=state),
        state=state,
        pipeline=pipeline,
        recorder=recorder,
        services=ServerServices(
            timestamp_provider=timestamp_provider or _FixedTimestampProvider()
        ),
        sleep=sleep or _ImmediateSleep(),
    )
    return Harness(
        reader=stream_reader,
        writer=stream_writer,
        state=state,
        recorder=recorder,
        connection=connection,
        responded=responded,
    )


def _close_after(
    after_body_bytes: int, *, reset: bool = False
) -> HeaderMiddleware:
    async def middleware(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = (ctx, call_next)
        return CloseDuringRequest(
            after_body_bytes=after_body_bytes, reset=reset
        )

    return middleware


async def _continue(
    ctx: HeaderContext, call_next: HeaderNext
) -> HeaderDecision:
    _ = ctx
    return await call_next()


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_run_serves_request_then_records_client_close_on_eof() -> None:
    harness = _harness()
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert harness.written == JSON_RESPONSE
    assert closed.reason == "client"
    assert closed.phase == "idle"
    assert not closed.reset
    assert closed.client == CLIENT
    assert closed.requests_completed == 1
    assert closed.bytes_read == len(GET)
    assert closed.bytes_consumed == len(GET)
    assert closed.bytes_written == len(JSON_RESPONSE)
    assert closed.timestamp == TIMESTAMP
    assert len(harness.recorder.requests) == 1
    assert harness.recorder.exchanges[0].closed is None
    assert harness.writer.close.called


@pytest.mark.asyncio
async def test_run_serves_pipelined_requests_in_order() -> None:
    harness = _harness()
    harness.feed(GET + GET_TWO, eof=True)

    closed = await harness.run()

    assert harness.responded == ["/one", "/two"]
    assert harness.written == JSON_RESPONSE * 2
    assert closed.requests_completed == 2
    assert closed.bytes_consumed == len(GET) + len(GET_TWO)


@pytest.mark.asyncio
async def test_close_connection_records_close_response_writes_nothing() -> (
    None
):
    harness = _harness(response=CloseConnection())
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert harness.written == b""
    assert closed.reason == "close_response"
    assert closed.phase == "response"
    assert not closed.reset
    assert closed.requests_completed == 1
    exchange = harness.recorder.exchanges[0]
    assert exchange.response is None
    assert exchange.closed is closed
    assert harness.recorder.responses == []
    harness.writer.close.assert_called_once()
    harness.writer.transport.abort.assert_not_called()


@pytest.mark.asyncio
async def test_close_connection_delay_waits_on_injected_sleep() -> None:
    sleep = _ImmediateSleep()
    harness = _harness(response=CloseConnection(delay=0.25), sleep=sleep)
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert sleep.calls == [0.25]
    assert closed.reason == "close_response"


@pytest.mark.asyncio
async def test_close_connection_without_delay_never_sleeps() -> None:
    sleep = _ImmediateSleep()
    harness = _harness(response=CloseConnection(), sleep=sleep)
    harness.feed(GET, eof=True)

    await harness.run()

    assert sleep.calls == []


@pytest.mark.asyncio
async def test_close_connection_reset_arms_linger_and_aborts() -> None:
    sock = create_autospec(socket.socket, instance=True)
    harness = _harness(
        response=CloseConnection(reset=True), writer=_fake_writer(sock)
    )
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reset
    sock.setsockopt.assert_called_once_with(
        socket.SOL_SOCKET, socket.SO_LINGER, pack_linger_option()
    )
    harness.writer.transport.abort.assert_called_once()


@pytest.mark.asyncio
async def test_reset_without_socket_skips_linger_and_aborts() -> None:
    harness = _harness(response=CloseConnection(reset=True))
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reset
    harness.writer.transport.abort.assert_called_once()


@pytest.mark.asyncio
async def test_reset_setsockopt_failure_logs_warning_and_still_aborts(
    caplog: pytest.LogCaptureFixture,
) -> None:
    sock = create_autospec(socket.socket, instance=True)
    sock.setsockopt.side_effect = OSError("no linger here")
    harness = _harness(
        response=CloseConnection(reset=True), writer=_fake_writer(sock)
    )
    harness.feed(GET, eof=True)

    with caplog.at_level(logging.WARNING, logger="localstub.server"):
        closed = await harness.run()

    assert closed.reset
    harness.writer.transport.abort.assert_called_once()
    assert "Could not request a TCP reset" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_during_close_delay_records_shutdown() -> None:
    sleep = _HeldSleep()
    harness = _harness(response=CloseConnection(delay=0.5), sleep=sleep)
    harness.feed(GET)
    task = asyncio.create_task(harness.connection.run())

    await asyncio.wait_for(sleep.entered.wait(), timeout=1.0)
    harness.connection.shutdown()
    sleep.release.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.closed.reason == "shutdown"
    assert harness.closed.phase == "response"
    assert harness.recorder.exchanges[0].closed is harness.closed


@pytest.mark.asyncio
async def test_shutdown_interrupts_idle_wait_and_run_returns_normally() -> (
    None
):
    harness = _harness()
    task = asyncio.create_task(harness.connection.run())
    await _settle()

    harness.connection.shutdown()
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.closed.reason == "shutdown"
    assert harness.closed.phase == "idle"
    assert harness.closed.requests_completed == 0


@pytest.mark.asyncio
async def test_shutdown_is_a_no_op_after_run_finished() -> None:
    harness = _harness()
    harness.feed(GET, eof=True)
    closed = await harness.run()

    harness.connection.shutdown()

    assert harness.recorder.closed_connections == [closed]
    assert closed.reason == "client"


@pytest.mark.asyncio
async def test_cancelling_run_finalizes_once_and_reraises() -> None:
    harness = _harness()
    task = asyncio.create_task(harness.connection.run())
    await _settle()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert harness.closed.reason == "shutdown"
    harness.writer.transport.abort.assert_called_once()
    assert len(harness.recorder.closed_connections) == 1


@pytest.mark.asyncio
async def test_cancelling_run_during_wait_closed_aborts_transport() -> None:
    release = asyncio.Event()

    async def wait_closed() -> None:
        await release.wait()

    writer = _fake_writer()
    writer.wait_closed.side_effect = wait_closed
    writer.transport.abort.side_effect = release.set
    harness = _harness(writer=writer)
    harness.feed(GET, eof=True)
    task = asyncio.create_task(harness.connection.run())
    await _settle()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    writer.transport.abort.assert_called_once()
    assert harness.closed.reason == "client"


@pytest.mark.asyncio
async def test_shutdown_from_inside_loop_records_and_exits_after_request() -> (
    None
):
    harness = _harness(response=HTTPResponse.json({"ok": True}))

    def handler(ctx: ResponderContext) -> ResponseSpec:
        _ = ctx
        harness.connection.shutdown()
        assert harness.connection.owns_current_task()
        return HTTPResponse.json({"ok": True})

    harness = _harness(response=handler)
    harness.feed(GET + GET_TWO, eof=True)

    closed = await asyncio.wait_for(harness.run(), timeout=1.0)

    assert harness.written == JSON_RESPONSE
    assert harness.responded == ["/one"]
    assert closed.reason == "shutdown"
    assert closed.phase == "response"
    assert harness.recorder.exchanges[0].closed is closed


def test_owns_current_task_is_false_without_a_running_loop() -> None:
    harness = _harness(reader=_ScriptedReader([]))

    assert not harness.connection.owns_current_task()


@pytest.mark.asyncio
async def test_shutdown_before_run_records_shutdown_with_zero_counters() -> (
    None
):
    harness = _harness()
    harness.feed(GET)

    harness.connection.shutdown()
    closed = await harness.run()

    assert closed.reason == "shutdown"
    assert closed.phase == "idle"
    assert closed.bytes_read == 0
    assert harness.written == b""
    harness.writer.close.assert_called_once()


def test_finalize_is_idempotent_and_publishes_once() -> None:
    harness = _harness(reader=_ScriptedReader([]))

    harness.connection.finalize()
    harness.connection.finalize()

    assert len(harness.recorder.closed_connections) == 1
    assert harness.closed.reason == "error"


def test_decide_close_first_decision_wins_and_snapshots_counters() -> None:
    state = ConnectionState(client=CLIENT, bytes_read=7)

    first = state.decide_close(
        "client", "idle", reset=False, timestamp=TIMESTAMP
    )
    state.bytes_read = 99
    second = state.decide_close(
        "shutdown", "response", reset=True, timestamp=TIMESTAMP
    )

    assert second is first
    assert first.reason == "client"
    assert first.bytes_read == 7
    assert state.closed is first


@pytest.mark.asyncio
async def test_close_during_request_zero_threshold_records_partial() -> None:
    harness = _harness(header_middlewares=[_close_after(0)])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert not request.body_complete
    assert request.body == b""
    assert request.wire_raw_bytes == POST_HEAD
    assert closed.reason == "request_read"
    assert closed.phase == "request_body"
    assert closed.requests_completed == 0
    assert closed.bytes_consumed == len(POST_HEAD)
    assert closed.bytes_read == len(POST_HEAD) + 10
    assert harness.recorder.exchanges[0].closed is closed
    assert harness.recorder.exchanges[0].response is None
    assert harness.responded == []


@pytest.mark.asyncio
async def test_close_during_request_below_content_length_keeps_prefix() -> (
    None
):
    harness = _harness(header_middlewares=[_close_after(4)])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert not request.body_complete
    assert request.body == b"0123"
    assert request.wire_raw_bytes == POST_HEAD + b"0123"
    assert closed.phase == "request_body"


@pytest.mark.parametrize("threshold", [10, 11])
@pytest.mark.asyncio
async def test_close_during_request_at_or_above_length_completes_request(
    threshold: int,
) -> None:
    harness = _harness(header_middlewares=[_close_after(threshold)])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert request.body_complete
    assert request.body == b"0123456789"
    assert closed.reason == "request_read"
    assert closed.phase == "response"
    assert closed.requests_completed == 1
    assert harness.responded == []


@pytest.mark.asyncio
async def test_close_during_request_bodyless_request_closes_at_headers() -> (
    None
):
    harness = _harness(header_middlewares=[_close_after(100)])
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert harness.recorder.requests[0].body_complete
    assert closed.phase == "response"
    assert closed.requests_completed == 1


@pytest.mark.asyncio
async def test_close_during_request_upgrade_no_body_closes_at_headers() -> (
    None
):
    harness = _harness(header_middlewares=[_close_after(0)])
    harness.feed(UPGRADE_HEAD, eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert request.body_complete
    assert request.body == b""
    assert request.wire_raw_bytes == UPGRADE_HEAD
    assert closed.reason == "request_read"
    assert closed.phase == "response"
    assert closed.requests_completed == 1


@pytest.mark.asyncio
async def test_close_during_request_peer_eof_before_declared_length() -> None:
    harness = _harness(header_middlewares=[_close_after(8)])
    harness.feed(POST_HEAD + b"012", eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert not request.body_complete
    assert request.body == b"012"
    assert closed.reason == "request_read"
    assert closed.phase == "request_body"


@pytest.mark.asyncio
async def test_close_during_request_reset_flag_aborts() -> None:
    harness = _harness(header_middlewares=[_close_after(0, reset=True)])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    assert closed.reset
    harness.writer.transport.abort.assert_called_once()


@pytest.mark.asyncio
async def test_close_during_request_chunked_stops_within_chunk() -> None:
    harness = _harness(header_middlewares=[_close_after(7)])
    harness.feed(CHUNKED_HEAD + CHUNKED_BODY, eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert request.body == b"hello w"
    assert request.wire_raw_bytes == CHUNKED_HEAD + b"5\r\nhello\r\n6\r\n w"
    assert not request.body_complete
    assert closed.bytes_consumed == len(request.wire_raw_bytes)
    assert closed.bytes_read == len(CHUNKED_HEAD) + len(CHUNKED_BODY)


@pytest.mark.asyncio
async def test_close_during_request_chunked_stops_at_chunk_boundary() -> None:
    harness = _harness(header_middlewares=[_close_after(5)])
    harness.feed(CHUNKED_HEAD + CHUNKED_BODY, eof=True)

    await harness.run()

    request = harness.recorder.requests[0]
    assert request.body == b"hello"
    assert request.wire_raw_bytes == CHUNKED_HEAD + b"5\r\nhello\r\n"
    assert not request.body_complete


@pytest.mark.parametrize("threshold", [11, 12])
@pytest.mark.asyncio
async def test_close_during_request_chunked_body_within_threshold_completes(
    threshold: int,
) -> None:
    harness = _harness(header_middlewares=[_close_after(threshold)])
    harness.feed(CHUNKED_HEAD + CHUNKED_BODY, eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert request.body == b"hello world"
    assert request.body_complete
    assert request.wire_raw_bytes == CHUNKED_HEAD + CHUNKED_BODY
    assert closed.phase == "response"


@pytest.mark.asyncio
async def test_close_during_request_chunked_split_reads_with_trailers() -> (
    None
):
    harness = _harness(header_middlewares=[_close_after(6)])
    harness.feed(CHUNKED_HEAD + b"5;ext=1\r\nhel")
    task = asyncio.create_task(harness.connection.run())
    await _settle()

    assert harness.recorder.requests == []
    harness.feed(b"lo\r\n3\r\nabc\r\n0\r\nX-Trailer: v\r\n\r\n", eof=True)
    await asyncio.wait_for(task, timeout=1.0)

    request = harness.recorder.requests[0]
    assert request.body == b"helloa"
    assert request.wire_raw_bytes == (
        CHUNKED_HEAD + b"5;ext=1\r\nhello\r\n3\r\na"
    )
    assert not request.body_complete


@pytest.mark.asyncio
async def test_close_during_request_short_body_excludes_pipelined() -> None:
    harness = _harness(header_middlewares=[_close_after(100)])
    harness.feed(CHUNKED_HEAD + CHUNKED_BODY + GET_TWO, eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert request.body_complete
    assert request.wire_raw_bytes == CHUNKED_HEAD + CHUNKED_BODY
    assert len(harness.recorder.requests) == 1
    assert closed.bytes_read - closed.bytes_consumed == len(GET_TWO)


@pytest.mark.asyncio
async def test_header_middleware_false_closes_without_consuming_body() -> None:
    async def refuse(ctx: HeaderContext, call_next: HeaderNext) -> bool:
        _ = (ctx, call_next)
        return False

    harness = _harness(header_middlewares=[refuse])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    assert closed.reason == "request_read"
    assert closed.phase == "request_body"
    assert harness.recorder.requests[0].body == b""
    assert not closed.reset


@pytest.mark.asyncio
async def test_header_middleware_delegation_preserves_decision() -> None:
    harness = _harness(header_middlewares=[_continue, _close_after(3)])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    await harness.run()

    assert harness.recorder.requests[0].body == b"012"


@pytest.mark.asyncio
async def test_header_middleware_unknown_decision_records_error() -> None:
    async def bogus(ctx: HeaderContext, call_next: HeaderNext) -> Any:
        _ = (ctx, call_next)
        return None

    harness = _harness(header_middlewares=[bogus])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    assert closed.reason == "error"
    assert closed.phase == "request_body"
    request = harness.recorder.requests[0]
    assert not request.body_complete
    assert request.body == b""
    assert harness.recorder.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_idle_timeout_fires_when_timer_wins() -> None:
    sleep = _ImmediateSleep()
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.2), sleep=sleep)
    harness.feed(GET)

    closed = await asyncio.wait_for(harness.run(), timeout=1.0)

    assert sleep.calls == [0.2]
    assert closed.reason == "idle_timeout"
    assert closed.phase == "idle"
    assert not closed.reset
    assert closed.requests_completed == 1
    assert harness.recorder.exchanges[0].closed is None
    harness.writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_idle_timeout_reset_flag_aborts() -> None:
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.2, reset=True))
    harness.feed(GET)

    closed = await asyncio.wait_for(harness.run(), timeout=1.0)

    assert closed.reason == "idle_timeout"
    assert closed.reset
    harness.writer.transport.abort.assert_called_once()


@pytest.mark.parametrize(
    ("build", "data", "reason"),
    [
        (
            lambda writer: _harness(
                keep_alive=KeepAlivePolicy(timeout=0.2), writer=writer
            ),
            GET,
            "idle_timeout",
        ),
        (
            lambda writer: _harness(response=CloseConnection(), writer=writer),
            GET,
            "close_response",
        ),
        (
            lambda writer: _harness(
                header_middlewares=[_close_after(0)], writer=writer
            ),
            POST_HEAD + b"0123456789",
            "request_read",
        ),
        (
            lambda writer: _harness(
                response=HTTPResponse.text("abcdef"),
                transmission=FaultyTransmission([
                    DropConnection(after_bytes=2)
                ]),
                writer=writer,
            ),
            GET,
            "response_aborted",
        ),
    ],
)
@pytest.mark.asyncio
async def test_close_event_is_published_before_transport_close_completes(
    build: Callable[[Mock], Harness],
    data: bytes,
    reason: str,
) -> None:
    writer, release = _held_close_writer()
    harness = build(writer)
    harness.feed(data)
    task = asyncio.create_task(harness.connection.run())
    try:
        closed = await harness.recorder.next_closed_connection(timeout=1.0)
        still_running = not task.done()
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=1.0)

    assert still_running
    assert closed.reason == reason
    assert harness.recorder.closed_connections == [closed]


@pytest.mark.asyncio
async def test_shutdown_after_loop_finished_aborts_pending_close() -> None:
    writer, release = _held_close_writer()
    writer.transport.abort.side_effect = release.set
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.2), writer=writer)
    harness.feed(GET)
    task = asyncio.create_task(harness.connection.run())
    try:
        closed = await harness.recorder.next_closed_connection(timeout=1.0)
        harness.connection.shutdown()
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=1.0)

    writer.transport.abort.assert_called_once()
    assert closed.reason == "idle_timeout"
    assert harness.recorder.closed_connections == [closed]


@pytest.mark.asyncio
async def test_no_idle_timer_runs_before_the_first_request() -> None:
    sleep = _ImmediateSleep()
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.2), sleep=sleep)
    task = asyncio.create_task(harness.connection.run())
    await _settle()

    assert sleep.calls == []
    harness.feed(GET)
    await asyncio.wait_for(task, timeout=1.0)

    assert sleep.calls == [0.2]
    assert harness.closed.reason == "idle_timeout"


@pytest.mark.asyncio
async def test_idle_timer_disarmed_once_first_byte_arrives() -> None:
    sleep = _HeldSleep()
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.2), sleep=sleep)
    harness.feed(GET)
    task = asyncio.create_task(harness.connection.run())

    await asyncio.wait_for(sleep.entered.wait(), timeout=1.0)
    sleep.entered.clear()
    harness.feed(GET_TWO[:1])
    await _settle()
    assert not sleep.entered.is_set()
    harness.feed(GET_TWO[1:])
    await asyncio.wait_for(sleep.entered.wait(), timeout=1.0)
    harness.feed(eof=True)
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.responded == ["/one", "/two"]
    assert harness.closed.reason == "client"
    assert sleep.calls == [0.2, 0.2]


@pytest.mark.asyncio
async def test_idle_wait_reads_buffered_next_request_without_duplication() -> (
    None
):
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.2))
    harness.feed(GET + GET_TWO, eof=True)

    closed = await asyncio.wait_for(harness.run(), timeout=1.0)

    assert harness.responded == ["/one", "/two"]
    assert [r.wire_raw_bytes for r in harness.recorder.requests] == [
        GET,
        GET_TWO,
    ]
    assert closed.reason == "client"
    assert closed.bytes_read == len(GET) + len(GET_TWO)


@pytest.mark.asyncio
async def test_zero_timeout_closes_after_response_with_pipelined_request() -> (
    None
):
    sleep = _ImmediateSleep()
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.0), sleep=sleep)
    harness.feed(GET + GET_TWO, eof=True)

    closed = await harness.run()

    assert harness.responded == ["/one"]
    assert b"Connection: close" not in harness.written
    assert closed.reason == "idle_timeout"
    assert closed.phase == "after_response"
    assert closed.bytes_read - closed.bytes_consumed == len(GET_TWO)
    assert closed.bytes_written == len(JSON_RESPONSE)
    assert harness.recorder.exchanges[0].closed is closed
    assert sleep.calls == []


@pytest.mark.asyncio
async def test_max_requests_adds_connection_close_and_records_event() -> None:
    harness = _harness(keep_alive=KeepAlivePolicy(max_requests=2))
    harness.feed(GET + GET_TWO + GET, eof=True)

    closed = await harness.run()

    assert harness.responded == ["/one", "/two"]
    first, second = harness.recorder.responses
    assert b"Connection: close" not in first.wire_raw_bytes
    assert b"Connection: close\r\n" in second.wire_raw_bytes
    assert closed.reason == "max_requests"
    assert closed.phase == "after_response"
    assert closed.requests_completed == 2
    assert harness.recorder.exchanges[1].closed is closed
    assert harness.recorder.exchanges[0].closed is None


@pytest.mark.asyncio
async def test_max_requests_wins_over_connection_close_header() -> None:
    harness = _harness(keep_alive=KeepAlivePolicy(max_requests=1))
    harness.feed(
        b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
        eof=True,
    )

    closed = await harness.run()

    assert closed.reason == "max_requests"


@pytest.mark.asyncio
async def test_connection_close_request_header_records_connection_close() -> (
    None
):
    harness = _harness()
    harness.feed(
        b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
        + GET_TWO,
        eof=True,
    )

    closed = await harness.run()

    assert harness.responded == ["/"]
    assert b"Connection: close\r\n" in harness.written
    assert closed.reason == "connection_close"
    assert closed.phase == "after_response"
    assert harness.recorder.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_close_token_is_appended_to_existing_connection_header() -> None:
    harness = _harness(
        response=HTTPResponse.text("ok", headers={"Connection": "x-custom"}),
        keep_alive=KeepAlivePolicy(max_requests=1),
    )
    harness.feed(GET, eof=True)

    await harness.run()

    assert b"Connection: x-custom, close\r\n" in harness.written
    assert harness.written.count(b"Connection:") == 1


@pytest.mark.asyncio
async def test_keep_alive_hint_advertised_with_connection_token() -> None:
    harness = _harness(
        keep_alive=KeepAlivePolicy(timeout=5, max_requests=100, advertise=True)
    )
    harness.feed(GET, eof=True)

    await harness.run()

    assert b"Keep-Alive: timeout=5, max=100\r\n" in harness.written
    assert b"Connection: keep-alive\r\n" in harness.written


@pytest.mark.asyncio
async def test_keep_alive_hint_appends_token_to_existing_header() -> None:
    harness = _harness(
        response=HTTPResponse.text(
            "ok", headers={"Connection": "Keep-Alive, x-custom"}
        ),
        keep_alive=KeepAlivePolicy(
            timeout=0.5, max_requests=3, advertise=True
        ),
    )
    harness.feed(GET, eof=True)

    await harness.run()

    assert b"Keep-Alive: max=3\r\n" in harness.written
    assert b"Connection: Keep-Alive, x-custom\r\n" in harness.written


@pytest.mark.asyncio
async def test_keep_alive_hint_not_advertised_on_closing_response() -> None:
    harness = _harness(
        keep_alive=KeepAlivePolicy(timeout=5, max_requests=1, advertise=True)
    )
    harness.feed(GET, eof=True)

    await harness.run()

    assert b"Keep-Alive:" not in harness.written
    assert b"Connection: close\r\n" in harness.written


@pytest.mark.asyncio
async def test_keep_alive_hint_is_omitted_when_nothing_to_advertise() -> None:
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.5, advertise=True))
    harness.feed(GET, eof=True)

    await harness.run()

    assert b"Keep-Alive:" not in harness.written
    assert b"Connection:" not in harness.written


def test_keep_alive_policy_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="timeout"):
        KeepAlivePolicy(timeout=-1)
    with pytest.raises(ValueError, match="max_requests"):
        KeepAlivePolicy(max_requests=0)


def test_keep_alive_policy_hint_without_advertise_is_none() -> None:
    assert KeepAlivePolicy(timeout=5, max_requests=2).keep_alive_hint() is None


@pytest.mark.asyncio
async def test_drop_connection_records_response_aborted() -> None:
    harness = _harness(
        response=HTTPResponse.text("abcdef"),
        transmission=FaultyTransmission([DropConnection(after_bytes=2)]),
    )
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert harness.written.endswith(b"\r\n\r\nab")
    assert closed.reason == "response_aborted"
    assert closed.phase == "response_body"
    assert not closed.reset
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.wire_raw_bytes.endswith(b"\r\n\r\nab")
    assert exchange.response.body == b"abcdef"
    harness.writer.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
async def test_shutdown_during_drop_close_retains_truncated_response(
    reset: bool,
) -> None:
    closing = asyncio.Event()
    release = asyncio.Event()

    async def wait_closed() -> None:
        closing.set()
        await release.wait()

    writer = _fake_writer()
    writer.wait_closed.side_effect = wait_closed
    harness = _harness(
        response=HTTPResponse.text("abcdef"),
        transmission=FaultyTransmission([
            DropConnection(after_bytes=2, reset=reset)
        ]),
        writer=writer,
    )
    harness.feed(GET)
    task = asyncio.create_task(harness.connection.run())
    try:
        await asyncio.wait_for(closing.wait(), timeout=1.0)
        harness.connection.shutdown()
        release.set()
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        release.set()
        harness.connection.shutdown()
        await asyncio.wait_for(task, timeout=1.0)

    closed = harness.closed
    assert closed.reason == "response_aborted"
    assert closed.phase == "response_body"
    assert closed.reset == reset
    assert len(harness.recorder.exchanges) == 1
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.body == b"abcdef"
    assert exchange.response.wire_raw_bytes.endswith(b"\r\n\r\nab")
    assert exchange.response.wire_raw_bytes == harness.written
    assert closed.bytes_written == len(exchange.response.wire_raw_bytes)
    assert harness.recorder.responses == [exchange.response]


@pytest.mark.asyncio
async def test_drop_connection_reset_arms_linger() -> None:
    sock = create_autospec(socket.socket, instance=True)
    harness = _harness(
        response=HTTPResponse.text("abcdef"),
        transmission=FaultyTransmission([
            DropConnection(after_bytes=0, reset=True)
        ]),
        writer=_fake_writer(sock),
    )
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reset
    sock.setsockopt.assert_called_once()
    harness.writer.transport.abort.assert_called_once()


@pytest.mark.asyncio
async def test_drop_connection_wins_over_planned_max_requests() -> None:
    harness = _harness(
        response=HTTPResponse.text("abcdef"),
        transmission=FaultyTransmission([DropConnection(after_bytes=2)]),
        keep_alive=KeepAlivePolicy(max_requests=1),
    )
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "response_aborted"
    assert b"Connection: close\r\n" in harness.written


@pytest.mark.asyncio
async def test_drain_reset_records_client_reset_not_error() -> None:
    writer = _fake_writer()
    writer.drain.side_effect = ConnectionResetError("peer reset")
    harness = _harness(
        writer=writer, keep_alive=KeepAlivePolicy(max_requests=1)
    )
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "client"
    assert closed.reset
    assert closed.phase == "response_body"
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None


@pytest.mark.asyncio
async def test_drain_broken_pipe_records_client_without_reset() -> None:
    writer = _fake_writer()
    writer.drain.side_effect = BrokenPipeError("gone")
    harness = _harness(writer=writer)
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "client"
    assert not closed.reset


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "transmission",
    [ImmediateTransmission(), ThrottledTransmission(chunk_size=1, delay=0)],
    ids=["immediate", "throttled"],
)
@pytest.mark.parametrize(
    "error",
    [ConnectionResetError("reset"), BrokenPipeError("gone"), OSError("gone")],
)
async def test_response_head_write_failure_records_client(
    transmission: TransmissionStrategy,
    error: OSError,
) -> None:
    writer = _fake_writer()
    writer.write.side_effect = error
    harness = _harness(
        writer=writer,
        transmission=transmission,
        keep_alive=KeepAlivePolicy(max_requests=1),
    )
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "client"
    assert closed.reset == isinstance(error, ConnectionResetError)
    assert closed.phase == "response_body"
    assert closed.bytes_written == len(harness.written)
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.wire_raw_bytes == harness.written


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_site", ["write", "drain"])
@pytest.mark.parametrize(
    "error",
    [ConnectionResetError("reset"), BrokenPipeError("gone"), OSError("gone")],
)
async def test_strategy_transport_failure_records_client(
    failure_site: str,
    error: OSError,
) -> None:
    writer = _fake_writer()
    if failure_site == "write":
        writer.write.side_effect = [None, error]
    else:
        writer.drain.side_effect = error
    harness = _harness(
        writer=writer,
        transmission=ThrottledTransmission(chunk_size=1, delay=0),
    )
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "client"
    assert closed.reset == isinstance(error, ConnectionResetError)
    assert closed.phase == "response_body"
    assert closed.bytes_written == len(harness.written)
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.wire_raw_bytes == harness.written


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("strategy failed"),
        OSError("strategy failed"),
        ConnectionResetError("strategy failed"),
    ],
)
async def test_strategy_error_records_error_with_partial_response(
    error: Exception,
) -> None:
    harness = _harness(transmission=_RaisingTransmission(error))
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "error"
    assert not closed.reset
    assert closed.phase == "response_body"
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.wire_raw_bytes.endswith(b"\r\n\r\n")
    assert harness.recorder.responses == [exchange.response]


@pytest.mark.asyncio
async def test_handler_exception_records_error_without_response() -> None:
    def handler(ctx: ResponderContext) -> ResponseSpec:
        _ = ctx
        raise ValueError("boom")

    harness = _harness(response=handler)
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "error"
    assert closed.phase == "response"
    exchange = harness.recorder.exchanges[0]
    assert exchange.response is None
    assert exchange.closed is closed
    assert harness.written == b""


@pytest.mark.asyncio
async def test_shutdown_mid_response_records_exchange_with_shutdown() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def handler(ctx: ResponderContext) -> ResponseSpec:
        _ = ctx
        started.set()
        await release.wait()
        return HTTPResponse.text("late")

    harness = _harness(response=handler)
    harness.feed(GET)
    task = asyncio.create_task(harness.connection.run())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    harness.connection.shutdown()
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.closed.reason == "shutdown"
    assert harness.closed.phase == "response"
    exchange = harness.recorder.exchanges[0]
    assert exchange.response is None
    assert exchange.closed is harness.closed
    assert harness.written == b""


@pytest.mark.asyncio
async def test_shutdown_mid_upload_records_partial_request() -> None:
    harness = _harness()
    harness.feed(POST_HEAD + b"0123")
    task = asyncio.create_task(harness.connection.run())
    await _settle()

    harness.connection.shutdown()
    await asyncio.wait_for(task, timeout=1.0)

    request = harness.recorder.requests[0]
    assert request.body == b"0123"
    assert not request.body_complete
    assert harness.closed.reason == "shutdown"
    assert harness.closed.phase == "request_body"
    assert harness.recorder.exchanges[0].closed is harness.closed


@pytest.mark.asyncio
async def test_shutdown_mid_headers_records_nothing() -> None:
    harness = _harness()
    harness.feed(b"GET /partial HTTP/1.1\r\nHost:")
    task = asyncio.create_task(harness.connection.run())
    await _settle()

    harness.connection.shutdown()
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.recorder.requests == []
    assert harness.closed.reason == "shutdown"
    assert harness.closed.phase == "request_headers"


@pytest.mark.asyncio
async def test_policy_close_counters_include_response_bytes() -> None:
    harness = _harness(keep_alive=KeepAlivePolicy(max_requests=1))
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.bytes_written == len(harness.written)
    assert closed.bytes_consumed == len(GET)


@pytest.mark.asyncio
async def test_header_phase_interim_and_final_send_then_close() -> None:
    async def reject(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(HTTPResponse(status=100))
        await ctx.send(HTTPResponse.text("Forbidden", status=403))
        return CloseDuringRequest()

    harness = _harness(header_middlewares=[reject])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    exchange = harness.recorder.exchanges[0]
    assert [r.status for r in exchange.interim_responses] == [100]
    assert exchange.interim_responses[0].wire_raw_bytes == (
        b"HTTP/1.1 100 Continue\r\n\r\n"
    )
    assert exchange.response is not None
    assert exchange.response.status == 403
    assert exchange.response.wire_raw_bytes.endswith(b"Forbidden")
    assert exchange.closed is closed
    assert not exchange.request.body_complete
    assert closed.reason == "request_read"
    assert harness.recorder.responses == [exchange.response]
    assert harness.written == (
        b"HTTP/1.1 100 Continue\r\n\r\n" + exchange.response.wire_raw_bytes
    )


@pytest.mark.asyncio
async def test_header_phase_final_send_then_true_skips_responder() -> None:
    async def answer(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        if ctx.headers.path == "/up":
            await ctx.send(HTTPResponse.text("early", status=202))
        return True

    harness = _harness(header_middlewares=[answer])
    harness.feed(POST_HEAD + b"0123456789" + GET_TWO, eof=True)

    closed = await harness.run()

    assert harness.responded == ["/two"]
    first, second = harness.recorder.exchanges
    assert first.request.body_complete
    assert first.request.body == b"0123456789"
    assert first.response is not None
    assert first.response.status == 202
    assert first.closed is None
    assert second.response is not None
    assert second.response.status == 200
    assert closed.reason == "client"
    assert closed.requests_completed == 2


@pytest.mark.asyncio
async def test_header_phase_final_send_prepares_policy_close() -> None:
    async def answer(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(HTTPResponse.text("early", status=202))
        return True

    harness = _harness(
        header_middlewares=[answer], keep_alive=KeepAlivePolicy(max_requests=1)
    )
    harness.feed(POST_HEAD + b"0123456789" + GET_TWO, eof=True)

    closed = await harness.run()

    assert harness.responded == []
    assert b"Connection: close\r\n" in harness.written
    assert closed.reason == "max_requests"
    assert closed.phase == "after_response"
    assert harness.recorder.exchanges[0].closed is closed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [HTTPResponse(status=100), HTTPResponse.text("Forbidden", status=403)],
)
@pytest.mark.parametrize(
    "error",
    [ConnectionResetError("reset"), BrokenPipeError("gone"), OSError("gone")],
)
async def test_header_phase_write_failure_records_client(
    response: HTTPResponse,
    error: OSError,
) -> None:
    async def answer(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(response)
        return True

    writer = _fake_writer()
    writer.write.side_effect = error
    harness = _harness(header_middlewares=[answer], writer=writer)
    harness.feed(POST_HEAD, eof=True)

    closed = await harness.run()

    assert closed.reason == "client"
    assert closed.phase == "request_body"
    assert closed.reset == isinstance(error, ConnectionResetError)
    assert closed.bytes_written == len(harness.written)
    assert harness.recorder.exchanges[0].closed is closed
    assert not harness.responded


@pytest.mark.asyncio
async def test_header_phase_second_final_send_raises() -> None:
    errors: list[Exception] = []

    async def twice(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(HTTPResponse.text("one", status=403))
        try:
            await ctx.send(HTTPResponse.text("two", status=404))
        except RuntimeError as exc:
            errors.append(exc)
            raise
        return False

    harness = _harness(header_middlewares=[twice])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    assert len(errors) == 1
    assert "already sent" in str(errors[0])
    assert closed.reason == "error"
    assert closed.phase == "request_body"
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.status == 403


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [HTTPResponse(status=100), HTTPResponse.text("Forbidden", status=403)],
)
@pytest.mark.parametrize(
    "drain_error",
    [ConnectionResetError("peer reset"), asyncio.CancelledError()],
    ids=["reset", "cancelled"],
)
async def test_header_phase_send_drain_failure_preserves_response(
    response: HTTPResponse,
    drain_error: BaseException,
) -> None:
    async def answer(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(response)
        return True

    writer = _fake_writer()
    writer.drain.side_effect = drain_error
    harness = _harness(header_middlewares=[answer], writer=writer)
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    if isinstance(drain_error, ConnectionResetError):
        assert closed.reason == "client"
        assert closed.reset
    else:
        assert closed.reason == "shutdown"
        assert not closed.reset
    assert closed.phase == "request_body"
    assert len(harness.recorder.exchanges) == 1
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is closed
    assert not exchange.request.body_complete
    if response.status == 100:
        assert exchange.response is None
        assert len(exchange.interim_responses) == 1
        recorded = exchange.interim_responses[0]
        assert recorded.wire_raw_bytes == b"HTTP/1.1 100 Continue\r\n\r\n"
        assert harness.recorder.responses == []
    else:
        assert exchange.interim_responses == ()
        assert exchange.response is not None
        recorded = exchange.response
        assert recorded.wire_raw_bytes.endswith(b"\r\n\r\nForbidden")
        assert harness.recorder.responses == [recorded]
    assert recorded.status == response.status
    assert recorded.wire_raw_bytes == harness.written
    assert closed.bytes_written == len(recorded.wire_raw_bytes)


@pytest.mark.asyncio
async def test_local_101_response_closes_and_ignores_later_bytes() -> None:
    harness = _harness(
        response=HTTPResponse(
            status=101,
            headers=Headers.from_items([
                ("Upgrade", "custom"),
                ("Connection", "Upgrade"),
            ]),
        )
    )
    harness.feed(GET + b"\x16\x03\x01not http", eof=True)

    closed = await harness.run()

    assert harness.responded == ["/one"]
    assert closed.reason == "connection_close"
    assert closed.phase == "after_response"
    assert len(harness.recorder.requests) == 1


@pytest.mark.asyncio
async def test_protocol_error_in_headers_records_no_request() -> None:
    harness = _harness()
    harness.feed(b"\x00\x01garbage", eof=True)

    closed = await harness.run()

    assert harness.recorder.requests == []
    assert closed.reason == "protocol_error"
    assert closed.phase == "request_headers"


@pytest.mark.asyncio
async def test_protocol_error_in_body_records_partial_request() -> None:
    harness = _harness()
    harness.feed(CHUNKED_HEAD + b"5\r\nhello\r\nZ\r\nbroken", eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert not request.body_complete
    assert request.body == b"hello"
    assert closed.reason == "protocol_error"
    assert closed.phase == "request_body"
    assert harness.recorder.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_client_eof_mid_body_records_partial_request() -> None:
    harness = _harness()
    harness.feed(POST_HEAD + b"abc", eof=True)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert request.body == b"abc"
    assert not request.body_complete
    assert closed.reason == "client"
    assert closed.phase == "request_body"
    assert not closed.reset
    assert closed.requests_completed == 0


@pytest.mark.asyncio
async def test_client_eof_mid_headers_records_client_request_headers() -> None:
    harness = _harness()
    harness.feed(b"GET / HTTP/1.1\r\nHost:", eof=True)

    closed = await harness.run()

    assert harness.recorder.requests == []
    assert closed.reason == "client"
    assert closed.phase == "request_headers"
    assert closed.bytes_read == len(b"GET / HTTP/1.1\r\nHost:")


@pytest.mark.asyncio
async def test_read_error_during_body_records_client_reset() -> None:
    reader = _ScriptedReader([POST_HEAD, ConnectionResetError("reset")])
    harness = _harness(reader=reader)

    closed = await harness.run()

    request = harness.recorder.requests[0]
    assert not request.body_complete
    assert closed.reason == "client"
    assert closed.reset
    assert closed.phase == "request_body"


@pytest.mark.asyncio
async def test_read_error_while_idle_records_client() -> None:
    reader = _ScriptedReader([GET, OSError("socket gone")])
    harness = _harness(reader=reader)

    closed = await harness.run()

    assert closed.reason == "client"
    assert not closed.reset
    assert closed.phase == "idle"
    assert closed.requests_completed == 1


@pytest.mark.asyncio
async def test_read_error_during_idle_race_records_client() -> None:
    sleep = _HeldSleep()
    reader = _ScriptedReader([GET, ConnectionResetError("reset")])
    harness = _harness(
        reader=reader, keep_alive=KeepAlivePolicy(timeout=1.0), sleep=sleep
    )

    closed = await asyncio.wait_for(harness.run(), timeout=1.0)

    assert closed.reason == "client"
    assert closed.reset
    assert closed.phase == "idle"


def _relayed(wire: bytes, *, status: int = 200) -> ForwardResult:
    return ForwardResult(
        status=status,
        reason="OK",
        headers=Message(),
        body=b"",
        wire_bytes=wire,
        relayed=RelayedResponse(
            status=status,
            headers=Message(),
            http_version="1.1",
            is_eof_delimited=False,
        ),
    )


def _forward_response(forwarder: Mock) -> ForwardProxyResponse:
    return ForwardProxyResponse(
        host="upstream",
        port=80,
        upstream_tls=False,
        request_wire_bytes=GET,
        request_method="GET",
        forwarder=forwarder,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_site", ["write", "drain"])
@pytest.mark.parametrize(
    "error", [ConnectionResetError("reset"), BrokenPipeError("gone")]
)
async def test_relay_client_transport_failure_records_client_response_body(
    failure_site: str,
    error: OSError,
) -> None:
    wire = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"

    async def relay(**kwargs: Any) -> ForwardResult:
        kwargs["client_writer"].write(wire)
        await kwargs["client_writer"].drain()
        return _relayed(wire)

    forwarder = Mock(spec=RawForwarder)
    forwarder.forward_and_relay = AsyncMock(side_effect=relay)
    writer = _fake_writer()
    if failure_site == "write":
        writer.write.side_effect = error
    else:
        writer.drain.side_effect = error
    harness = _harness(response=_forward_response(forwarder), writer=writer)
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "client"
    assert closed.phase == "response_body"
    assert closed.reset == isinstance(error, ConnectionResetError)
    assert closed.bytes_written == len(wire)
    assert len(harness.recorder.exchanges) == 1
    assert harness.recorder.exchanges[0].closed is closed


@pytest.mark.parametrize("write_response", [False, True])
@pytest.mark.asyncio
async def test_relay_upstream_failure_is_not_classified_as_client(
    write_response: bool,
) -> None:
    async def relay(**kwargs: Any) -> ForwardResult:
        if write_response:
            kwargs["client_writer"].write(b"HTTP/1.1 100 Continue\r\n\r\n")
        raise ConnectionResetError("upstream reset")

    forwarder = Mock(spec=RawForwarder)
    forwarder.forward_and_relay = AsyncMock(side_effect=relay)
    harness = _harness(response=_forward_response(forwarder))
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "error"
    assert not closed.reset
    assert closed.phase == ("response_body" if write_response else "response")


@pytest.mark.asyncio
async def test_relayed_response_counts_toward_max_requests_unmodified() -> (
    None
):
    relayed_wire = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"

    async def relay(**kwargs: Any) -> ForwardResult:
        kwargs["client_writer"].write(relayed_wire)
        await kwargs["client_writer"].drain()
        return _relayed(relayed_wire)

    forwarder = Mock(spec=RawForwarder)
    forwarder.forward_and_relay = AsyncMock(side_effect=relay)
    harness = _harness(
        response=_forward_response(forwarder),
        keep_alive=KeepAlivePolicy(max_requests=1),
    )
    harness.feed(GET + GET_TWO, eof=True)

    closed = await harness.run()

    assert harness.written == relayed_wire
    assert closed.reason == "max_requests"
    assert closed.requests_completed == 1
    assert harness.recorder.responses[0].wire_raw_bytes == relayed_wire


@pytest.mark.asyncio
async def test_relayed_101_records_connection_close() -> None:
    relayed_wire = b"HTTP/1.1 101 Switching Protocols\r\n\r\n"

    async def relay(**kwargs: Any) -> ForwardResult:
        kwargs["client_writer"].write(relayed_wire)
        return _relayed(relayed_wire, status=101)

    forwarder = Mock(spec=RawForwarder)
    forwarder.forward_and_relay = AsyncMock(side_effect=relay)
    harness = _harness(response=_forward_response(forwarder))
    harness.feed(GET + b"\x00tunnel bytes", eof=True)

    closed = await harness.run()

    assert closed.reason == "connection_close"
    assert closed.phase == "after_response"


@pytest.mark.asyncio
async def test_relay_closed_by_forwarder_records_response_aborted() -> None:
    async def relay(**kwargs: Any) -> ForwardResult:
        kwargs["client_writer"].close()
        await kwargs["client_writer"].wait_closed()
        return _relayed(b"")

    forwarder = Mock(spec=RawForwarder)
    forwarder.forward_and_relay = AsyncMock(side_effect=relay)
    harness = _harness(response=_forward_response(forwarder))
    harness.feed(GET + GET_TWO, eof=True)

    closed = await harness.run()

    assert harness.responded == ["/one"]
    assert closed.reason == "response_aborted"
    assert closed.phase == "response_body"
    assert not closed.reset


@pytest.mark.asyncio
async def test_relay_failure_falls_back_to_bad_gateway() -> None:
    forwarder = Mock(spec=RawForwarder)
    forwarder.forward_and_relay = AsyncMock(return_value=None)
    harness = _harness(response=_forward_response(forwarder))
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert harness.written.startswith(b"HTTP/1.1 502 Bad Gateway\r\n")
    assert closed.reason == "client"
    assert harness.recorder.responses[0].status == 502


@pytest.mark.asyncio
async def test_sender_middleware_observes_close_result() -> None:
    seen: list[SendResult] = []

    async def observe(
        ctx: SenderContext, response: ResponseSpec, call_next: SenderNext
    ) -> SendResult:
        _ = ctx
        result = await call_next(response)
        seen.append(result)
        return result

    harness = _harness(
        response=CloseConnection(reset=True), sender_middlewares=[observe]
    )
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert len(seen) == 1
    assert seen[0].recorded is None
    assert seen[0].should_close
    assert seen[0].closed is closed


@pytest.mark.asyncio
@pytest.mark.parametrize("forwarded", [False, True])
@pytest.mark.parametrize("reason", ["shutdown", "error"])
async def test_post_send_middleware_failure_records_after_response(
    forwarded: bool,
    reason: str,
) -> None:
    sent = asyncio.Event()
    release = asyncio.Event()

    async def post_send(
        ctx: SenderContext, response: ResponseSpec, call_next: SenderNext
    ) -> SendResult:
        _ = ctx
        await call_next(response)
        sent.set()
        await release.wait()
        raise RuntimeError("post-send failure")

    async def relay(**kwargs: Any) -> ForwardResult:
        kwargs["client_writer"].write(JSON_RESPONSE)
        await kwargs["client_writer"].drain()
        return _relayed(JSON_RESPONSE)

    response: ResponseSpec = DEFAULT_RESPONSE
    if forwarded:
        forwarder = Mock(spec=RawForwarder)
        forwarder.forward_and_relay = AsyncMock(side_effect=relay)
        response = _forward_response(forwarder)
    harness = _harness(response=response, sender_middlewares=[post_send])
    harness.feed(GET, eof=True)
    task = asyncio.create_task(harness.connection.run())
    await asyncio.wait_for(sent.wait(), timeout=1.0)

    assert harness.written == JSON_RESPONSE
    if reason == "shutdown":
        harness.connection.shutdown()
    else:
        release.set()
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.closed.reason == reason
    assert harness.closed.phase == "after_response"
    assert harness.recorder.exchanges[0].closed is harness.closed


@pytest.mark.asyncio
async def test_sender_middleware_replaces_response() -> None:
    async def replace(
        ctx: SenderContext, response: ResponseSpec, call_next: SenderNext
    ) -> SendResult:
        _ = (ctx, response)
        return await call_next(HTTPResponse.text("replaced"))

    harness = _harness(sender_middlewares=[replace])
    harness.feed(GET, eof=True)

    await harness.run()

    assert harness.written.endswith(b"replaced")
    assert harness.recorder.responses[0].body == b"replaced"


@pytest.mark.asyncio
async def test_head_request_suppresses_body_but_keeps_length() -> None:
    harness = _harness()
    harness.feed(b"HEAD /one HTTP/1.1\r\nHost: localhost\r\n\r\n", eof=True)

    await harness.run()

    assert harness.written.endswith(b"Content-Length: 12\r\n\r\n")


@pytest.mark.asyncio
async def test_bytes_written_counts_header_phase_sends() -> None:
    async def answer(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(HTTPResponse(status=100))
        return True

    harness = _harness(header_middlewares=[answer])
    harness.feed(POST_HEAD + b"0123456789", eof=True)

    closed = await harness.run()

    assert closed.bytes_written == len(harness.written)
    assert harness.written.startswith(b"HTTP/1.1 100 Continue\r\n\r\n")


@pytest.mark.asyncio
async def test_retained_buffers_receive_consumed_and_sent_bytes() -> None:
    received = BoundedByteBuffer(None)
    sent = BoundedByteBuffer(None)
    harness = _harness(received=received, sent=sent)
    harness.feed(GET + GET_TWO[:5], eof=True)

    closed = await harness.run()

    assert bytes(received) == GET + GET_TWO[:5]
    assert bytes(sent) == JSON_RESPONSE
    assert closed.bytes_consumed == len(GET) + 5


@pytest.mark.asyncio
async def test_counting_stream_reader_counts_bytes_from_stream() -> None:
    state = ConnectionState(client=None)
    reader = asyncio.StreamReader()
    reader.feed_data(b"abcdef")
    reader.feed_eof()
    counting = CountingStreamReader(reader, state)

    assert await counting.read(4) == b"abcd"
    assert await counting.read(4) == b"ef"
    assert await counting.read(4) == b""
    assert state.bytes_read == 6


@pytest.mark.asyncio
async def test_counting_stream_reader_drains_wrapped_read_ahead() -> None:
    state = ConnectionState(client=None)
    reader = asyncio.StreamReader()
    reader.feed_data(b"cdef")
    reader.feed_eof()
    unread_data(reader, b"ab")
    counting = CountingStreamReader(reader, state)

    assert await counting.read(4) == b"ab"
    assert await counting.read(4) == b"cdef"
    assert await counting.read(4) == b""
    assert state.bytes_read == 6


@pytest.mark.asyncio
async def test_counting_stream_reader_cancelled_read_counts_nothing() -> None:
    state = ConnectionState(client=None)
    reader = asyncio.StreamReader()
    counting = CountingStreamReader(reader, state)
    pending = asyncio.create_task(counting.read(4))
    await _settle()

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    reader.feed_data(b"late")

    assert state.bytes_read == 0
    assert await counting.read(4) == b"late"
    assert state.bytes_read == 4


def test_recording_writer_abort_delegates_to_transport() -> None:
    writer = _fake_writer()
    recording = RecordingStreamWriter(writer)

    recording.abort()

    writer.transport.abort.assert_called_once_with()
    writer.close.assert_not_called()


def test_recording_writer_feeds_state_bytes_written() -> None:
    state = ConnectionState(client=None)
    recording = RecordingStreamWriter(_fake_writer(), state=state)

    recording.write(b"abc")
    recording.write(b"")
    recording.writelines([b"de", b"f"])

    assert state.bytes_written == 6


def test_recording_writer_holds_only_current_response() -> None:
    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    recorder.start_response()
    recorder.write(b"first response")
    recorder.start_response()
    recorder.write(b"second response")

    assert recorder.bytes_sent == b"second response"


def test_recording_writer_reuses_single_write_through_empty_writes() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    recorder = RecordingStreamWriter(writer)
    body = b"x" * 10000

    assert recorder.bytes_sent == b""
    recorder.write(b"")
    recorder.write(body)
    recorder.write(b"")

    assert recorder.bytes_sent is body
    assert [call.args[0] for call in writer.write.call_args_list] == [
        b"",
        body,
        b"",
    ]


def test_recording_writer_coalesces_small_writes_around_large_write() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    recorder = RecordingStreamWriter(writer, coalesce_size=4)

    recorder.writelines([b"ab", b"c", b"defgh", b"i", b"jkl", b"m"])

    assert recorder.bytes_sent == b"abcdefghijklm"


def test_recording_writer_coalesce_size_below_one_raises_value_error() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)

    with pytest.raises(ValueError, match="coalesce_size"):
        RecordingStreamWriter(writer, coalesce_size=0)


@given(
    coalesce_size=st.integers(min_value=1, max_value=16),
    chunks=st.lists(st.binary(max_size=64), max_size=40),
)
def test_recording_writer_bytes_sent_matches_every_write(
    coalesce_size: int, chunks: list[bytes]
) -> None:
    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer, coalesce_size=coalesce_size)

    expected = b""
    for chunk in chunks:
        recorder.write(chunk)
        expected += chunk
        assert recorder.bytes_sent == expected

    assert [call.args[0] for call in writer.write.call_args_list] == chunks


def test_recording_writer_snapshots_survive_more_writes_and_reset() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    sent = BoundedByteBuffer(5)
    recorder = RecordingStreamWriter(writer, sent)
    recorder.write(b"ab")
    first = recorder.bytes_sent
    recorder.write(b"cd")
    second = recorder.bytes_sent
    recorder.writelines([b"", b"ef", b"gh"])
    third = recorder.bytes_sent
    recorder.start_response()

    assert recorder.bytes_sent == b""
    recorder.write(b"ij")
    assert first == b"ab"
    assert second == b"abcd"
    assert third == b"abcdefgh"
    assert recorder.bytes_sent == b"ij"
    assert bytes(sent) == b"fghij"
    assert sent.dropped == 5
    assert [call.args[0] for call in writer.write.call_args_list] == [
        b"ab",
        b"cd",
        b"",
        b"ef",
        b"gh",
        b"ij",
    ]


@pytest.mark.parametrize("prefix", [b"", b"previous"])
def test_recording_writer_preserves_capture_on_write_failure(
    prefix: bytes,
) -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    sent = BoundedByteBuffer(100)
    recorder = RecordingStreamWriter(writer, sent)
    recorder.write(prefix)
    writer.write.side_effect = ConnectionError("closed")

    with pytest.raises(ConnectionError, match="closed"):
        recorder.write(b"failed")

    assert recorder.bytes_sent == prefix + b"failed"
    assert bytes(sent) == prefix + b"failed"


def test_recording_writer_write_eof_delegates_to_wrapped_writer() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    recorder = RecordingStreamWriter(writer)

    recorder.write_eof()

    writer.write_eof.assert_called_once_with()


@pytest.mark.asyncio
async def test_recording_writer_wait_closed_awaits_wrapped_writer() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    recorder = RecordingStreamWriter(writer)

    await recorder.wait_closed()

    writer.wait_closed.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_recording_writer_cancelled_wait_keeps_shared_waiter() -> None:
    shared = asyncio.get_running_loop().create_future()

    async def wait_closed() -> None:
        await shared

    writer = create_autospec(asyncio.StreamWriter, instance=True)
    writer.wait_closed.side_effect = wait_closed
    recorder = RecordingStreamWriter(writer)
    waiting = asyncio.create_task(recorder.wait_closed())
    await asyncio.sleep(0)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert not shared.cancelled()
    shared.set_result(None)
    await asyncio.sleep(0)


def test_pack_linger_option_uses_shorts_on_windows() -> None:
    assert len(pack_linger_option(platform="win32")) == 4


def test_pack_linger_option_uses_ints_elsewhere() -> None:
    assert len(pack_linger_option(platform="linux")) == 8
    assert len(pack_linger_option(platform="darwin")) == 8


@pytest.mark.asyncio
async def test_str_body_is_utf8_encoded_with_matching_content_length() -> None:
    harness = _harness(response=HTTPResponse(status=200, body="héllo"))
    harness.feed(GET, eof=True)

    await harness.run()

    assert harness.written.endswith(b"Content-Length: 6\r\n\r\nh\xc3\xa9llo")
    assert harness.recorder.responses[0].body == "héllo".encode()


@pytest.mark.asyncio
async def test_close_decided_before_first_loop_turn_serves_nothing() -> None:
    harness = _harness()
    harness.feed(GET)
    task = asyncio.create_task(harness.connection.run())
    await asyncio.sleep(0)

    harness.connection.state.decide_close(
        "client", "idle", reset=False, timestamp=TIMESTAMP
    )
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.responded == []
    assert harness.closed.reason == "client"
    assert harness.connection.state.closed is harness.closed
    harness.writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_cancelling_run_while_finalizing_aborts_and_reraises() -> None:
    release = asyncio.Event()

    async def wait_closed() -> None:
        await release.wait()

    writer = _fake_writer()
    writer.wait_closed.side_effect = wait_closed
    writer.transport.abort.side_effect = release.set
    harness = _harness(writer=writer)
    harness.state.decide_close(
        "client", "idle", reset=False, timestamp=TIMESTAMP
    )
    task = asyncio.create_task(harness.connection.run())
    await _settle()

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    writer.transport.abort.assert_called_once()
    assert harness.closed.reason == "client"


@pytest.mark.asyncio
async def test_finalize_on_shutdown_closes_then_aborts_without_waiting() -> (
    None
):
    release = asyncio.Event()

    async def wait_closed() -> None:
        await release.wait()

    writer = _fake_writer()
    writer.wait_closed.side_effect = wait_closed
    writer.transport.abort.side_effect = release.set
    harness = _harness(writer=writer)
    harness.connection.shutdown()

    closed = await asyncio.wait_for(harness.run(), timeout=1.0)

    writer.close.assert_called_once()
    writer.transport.abort.assert_called_once()
    assert closed.reason == "shutdown"


def test_finalize_closes_writer_when_timestamp_provider_raises() -> None:
    writer = _fake_writer()
    harness = _harness(
        writer=writer,
        reader=_ScriptedReader([]),
        timestamp_provider=_RaisingTimestampProvider(),
    )

    with pytest.raises(RuntimeError, match="no timestamps"):
        harness.connection.finalize()

    writer.close.assert_called_once()
    assert harness.recorder.closed_connections == []


@pytest.mark.asyncio
async def test_finalize_logs_wait_closed_failure_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer = _fake_writer()
    writer.wait_closed.side_effect = OSError("already gone")
    harness = _harness(writer=writer)
    harness.connection.shutdown()

    with caplog.at_level(logging.DEBUG, logger="localstub.server"):
        closed = await harness.run()

    assert closed.reason == "shutdown"
    assert "Failed to close client writer" in caplog.text


@pytest.mark.asyncio
async def test_close_writer_logs_wait_closed_failure_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer = _fake_writer()
    writer.wait_closed.side_effect = OSError("already gone")
    harness = _harness(response=CloseConnection(), writer=writer)
    harness.feed(GET, eof=True)

    with caplog.at_level(logging.DEBUG, logger="localstub.server"):
        closed = await harness.run()

    assert closed.reason == "close_response"
    assert "Failed to close client writer" in caplog.text


@pytest.mark.asyncio
async def test_shutdown_during_idle_race_cancels_timer_and_records_idle() -> (
    None
):
    sleep = _HeldSleep()
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.2), sleep=sleep)
    harness.feed(GET)
    task = asyncio.create_task(harness.connection.run())
    await asyncio.wait_for(sleep.entered.wait(), timeout=1.0)

    harness.connection.shutdown()
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.closed.reason == "shutdown"
    assert harness.closed.phase == "idle"
    assert harness.closed.requests_completed == 1
    assert harness.recorder.exchanges[0].closed is None
    assert sleep.calls == [0.2]


@pytest.mark.asyncio
async def test_shutdown_during_header_middleware_records_whole_request() -> (
    None
):
    started = asyncio.Event()

    async def stall(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = ctx
        started.set()
        await asyncio.Event().wait()
        return await call_next()

    harness = _harness(header_middlewares=[stall])
    harness.feed(GET)
    task = asyncio.create_task(harness.connection.run())
    await asyncio.wait_for(started.wait(), timeout=1.0)

    harness.connection.shutdown()
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.recorder.requests[0].body_complete
    assert harness.closed.reason == "shutdown"
    assert harness.closed.phase == "response"
    assert harness.closed.requests_completed == 1
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is harness.closed
    assert exchange.response is None


@pytest.mark.asyncio
async def test_drain_failure_wins_over_planned_zero_timeout_close() -> None:
    writer = _fake_writer()
    writer.drain.side_effect = BrokenPipeError("gone")
    harness = _harness(writer=writer, keep_alive=KeepAlivePolicy(timeout=0.0))
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "client"
    assert closed.phase == "response_body"
    assert not closed.reset
    assert harness.recorder.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_drop_connection_wins_over_planned_zero_timeout_close() -> None:
    harness = _harness(
        response=HTTPResponse.text("abcdef"),
        transmission=FaultyTransmission([DropConnection(after_bytes=2)]),
        keep_alive=KeepAlivePolicy(timeout=0.0, reset=True),
    )
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "response_aborted"
    assert closed.phase == "response_body"
    assert not closed.reset


@pytest.mark.asyncio
async def test_first_byte_probe_counts_bytes_once_after_unread() -> None:
    sleep = _HeldSleep()
    harness = _harness(keep_alive=KeepAlivePolicy(timeout=0.2), sleep=sleep)
    harness.feed(GET)
    task = asyncio.create_task(harness.connection.run())
    await asyncio.wait_for(sleep.entered.wait(), timeout=1.0)

    harness.feed(GET_TWO[:1])
    await _settle()
    harness.feed(GET_TWO[1:], eof=True)
    await asyncio.wait_for(task, timeout=1.0)

    assert harness.responded == ["/one", "/two"]
    assert harness.closed.bytes_read == len(GET) + len(GET_TWO)
    assert harness.closed.bytes_consumed == len(GET) + len(GET_TWO)
    assert harness.recorder.requests[1].wire_raw_bytes == GET_TWO


@pytest.mark.asyncio
async def test_no_timeout_never_calls_sleep_between_requests() -> None:
    sleep = _ImmediateSleep()
    harness = _harness(sleep=sleep)
    harness.feed(GET + GET_TWO, eof=True)

    await harness.run()

    assert harness.responded == ["/one", "/two"]
    assert sleep.calls == []


@pytest.mark.asyncio
async def test_header_middleware_exception_records_request_with_error() -> (
    None
):
    async def explode(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = (ctx, call_next)
        raise ValueError("boom")

    harness = _harness(header_middlewares=[explode])
    harness.feed(GET, eof=True)

    closed = await harness.run()

    assert closed.reason == "error"
    assert closed.phase == "response"
    assert closed.requests_completed == 1
    assert harness.recorder.requests[0].body_complete
    exchange = harness.recorder.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is None
    assert harness.written == b""


@pytest.mark.asyncio
async def test_header_phase_final_send_counts_bodyless_request_once() -> None:
    async def answer(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(HTTPResponse.text("early", status=202))
        return True

    harness = _harness(
        header_middlewares=[answer], keep_alive=KeepAlivePolicy(max_requests=2)
    )
    harness.feed(GET + GET_TWO, eof=True)

    closed = await harness.run()

    first, second = harness.recorder.responses
    assert b"Connection: close" not in first.wire_raw_bytes
    assert b"Connection: close\r\n" in second.wire_raw_bytes
    assert closed.reason == "max_requests"
    assert closed.requests_completed == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("request_head", "response", "expected"),
    [
        (
            HEAD_ONE,
            HTTPResponse.raw(b"BODY"),
            b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\n",
        ),
        (
            GET,
            HTTPResponse(status=204, body=b"BODY"),
            b"HTTP/1.1 204 No Content\r\n\r\n",
        ),
        (
            GET,
            HTTPResponse(status=304, body=b"BODY"),
            b"HTTP/1.1 304 Not Modified\r\nContent-Length: 4\r\n\r\n",
        ),
    ],
)
async def test_header_phase_bodyless_send_keeps_next_response_aligned(
    request_head: bytes,
    response: HTTPResponse,
    expected: bytes,
) -> None:
    async def answer(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        if ctx.headers.path == "/one":
            await ctx.send(response)
        return True

    harness = _harness(header_middlewares=[answer])
    harness.feed(request_head + GET_TWO, eof=True)

    closed = await harness.run()

    first, second = harness.recorder.responses
    assert first.wire_raw_bytes == expected
    assert second.wire_raw_bytes == JSON_RESPONSE
    assert harness.written == expected + JSON_RESPONSE
    assert harness.responded == ["/two"]
    assert closed.reason == "client"
    assert closed.requests_completed == 2


@pytest.mark.asyncio
async def test_header_phase_101_is_final_and_closes_connection() -> None:
    async def switch(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = call_next
        await ctx.send(
            HTTPResponse(
                status=101,
                headers=Headers.from_items([
                    ("Upgrade", "custom"),
                    ("Connection", "Upgrade"),
                ]),
            )
        )
        return True

    harness = _harness(header_middlewares=[switch])
    harness.feed(UPGRADE_HEAD + b"\x16\x03\x01not http", eof=True)

    closed = await harness.run()

    assert harness.responded == []
    exchange = harness.recorder.exchanges[0]
    assert exchange.interim_responses == ()
    assert exchange.response is not None
    assert exchange.response.status == 101
    assert harness.written == exchange.response.wire_raw_bytes
    assert harness.written.startswith(b"HTTP/1.1 101 Switching Protocols\r\n")
    assert harness.written.count(b"HTTP/1.1") == 1
    assert closed.reason == "connection_close"
    assert closed.phase == "after_response"
    assert closed.requests_completed == 1
    assert len(harness.recorder.requests) == 1


@pytest.mark.asyncio
async def test_rejected_transfer_encoding_records_protocol_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    seen: list[str | None] = []

    async def observe(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        seen.append(ctx.headers.path)
        return await call_next()

    harness = _harness(header_middlewares=[observe])
    harness.feed(GZIP_HEAD + b"body", eof=True)

    with caplog.at_level(logging.ERROR, logger="localstub.server"):
        closed = await harness.run()

    request = harness.recorder.requests[0]
    assert not request.body_complete
    assert request.wire_raw_bytes == GZIP_HEAD
    assert request.body is None
    assert seen == []
    assert harness.responded == []
    assert harness.written == b""
    assert closed.reason == "protocol_error"
    assert closed.phase == "request_body"
    assert closed.requests_completed == 0
    assert closed.bytes_consumed == len(GZIP_HEAD)
    assert harness.recorder.exchanges[0].closed is closed
    assert caplog.records == []


@pytest.mark.asyncio
@given(
    prefix_count=st.integers(min_value=0, max_value=3),
    cause=st.sampled_from([
        "max_requests",
        "close",
        "drop",
        "eof",
        "reset",
        "shutdown",
        "cancel",
    ]),
    reset=st.booleans(),
    interrupt_body=st.booleans(),
    cleanup=st.lists(st.booleans(), min_size=1, max_size=6),
    client_phase=st.sampled_from(["idle", "request_headers", "request_body"]),
)
async def test_competing_close_causes_preserve_one_terminal_snapshot(
    prefix_count: int,
    cause: str,
    reset: bool,
    interrupt_body: bool,
    cleanup: list[bool],
    client_phase: str,
) -> None:
    sleep = _HeldSleep()
    faults: list[FaultStep] = []
    requests = [
        f"GET /{index} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
        for index in range(prefix_count + 1)
    ]

    def respond(ctx: ResponderContext) -> ResponseSpec:
        if ctx.request.target == f"/{prefix_count}":
            if cause == "close":
                return CloseConnection(reset=reset)
            if cause == "drop":
                faults.append(DropConnection(after_bytes=2, reset=reset))
            if cause in {"shutdown", "cancel"}:
                if not interrupt_body:
                    return CloseConnection(delay=1, reset=reset)
                faults.extend([Delay(1), DropConnection(2, reset=reset)])
        return HTTPResponse.text("payload")

    partial = (
        {
            "idle": b"",
            "request_headers": b"GET /unfinished HTTP/1.1\r\n",
            "request_body": POST_HEAD + b"abc",
        }[client_phase]
        if cause in {"eof", "reset"}
        else b""
    )
    script: list[bytes | Exception] = [
        *requests,
        *([partial] if partial else []),
    ]
    if cause == "reset":
        script.append(ConnectionResetError())
    harness = _harness(
        reader=_ScriptedReader(script),
        response=respond,
        transmission=FaultyTransmission(faults, sleep=sleep),
        keep_alive=KeepAlivePolicy(
            max_requests=None if cause in {"eof", "reset"} else len(requests)
        ),
        sleep=sleep,
    )
    await _finish_close_history(harness, sleep, cause)

    closed = harness.closed
    reasons = {
        "max_requests": "max_requests",
        "close": "close_response",
        "drop": "response_aborted",
        "eof": "client",
        "reset": "client",
        "shutdown": "shutdown",
        "cancel": "shutdown",
    }
    phases = {
        "max_requests": "after_response",
        "close": "response",
        "drop": "response_body",
        "eof": client_phase,
        "reset": client_phase,
        "shutdown": "response_body" if interrupt_body else "response",
        "cancel": "response_body" if interrupt_body else "response",
    }
    assert closed.reason == reasons[cause]
    assert closed.phase == phases[cause]
    assert closed.reset == (
        cause == "reset" or (cause in {"close", "drop"} and reset)
    )
    assert closed.requests_completed == len(requests)
    assert closed.bytes_read == sum(map(len, requests)) + len(partial)
    assert closed.bytes_consumed == sum(map(len, requests)) + len(partial)
    assert closed.bytes_written == len(harness.written)
    snapshot = dataclasses.astuple(closed)
    harness.state.bytes_read += 1
    harness.state.bytes_consumed += 1
    harness.state.bytes_written += 1
    harness.state.requests_completed += 1
    for shutdown in cleanup:
        if shutdown:
            harness.connection.shutdown()
        else:
            harness.connection.finalize()
        assert harness.closed is closed
        assert dataclasses.astuple(closed) == snapshot
    with pytest.raises(dataclasses.FrozenInstanceError):
        closed.reason = "error"


@pytest.mark.asyncio
@given(
    cases=st.lists(
        st.tuples(
            st.sampled_from(["GET", "HEAD"]),
            st.sampled_from([200, 204, 304, 101]),
            st.lists(st.sampled_from([100, 102, 103]), max_size=4),
            st.booleans(),
            st.one_of(st.none(), st.integers(min_value=0, max_value=16)),
            st.binary(max_size=16),
        ),
        min_size=2,
        max_size=5,
    ),
)
@example(
    cases=[
        ("HEAD", 200, [100, 103], True, None, b"head"),
        ("GET", 204, [102, 103], False, None, b"no content"),
        ("GET", 304, [100], True, None, b"not modified"),
        ("GET", 200, [100, 102, 103], False, 3, b"truncated"),
        ("GET", 200, [], False, None, b"unreachable"),
    ]
)
@example(
    cases=[
        ("GET", 200, [100, 103], False, None, b"normal"),
        ("GET", 101, [103, 103], True, None, b"upgrade"),
        ("GET", 200, [], False, None, b"unreachable"),
    ]
)
async def test_recorded_exchanges_account_for_all_response_bytes_in_order(
    cases: list[tuple[str, int, list[int], bool, int | None, bytes]],
) -> None:
    faults: list[FaultStep] = []

    def response_for(index: int) -> HTTPResponse:
        _, status, _, _, _, body = cases[index]
        return HTTPResponse(
            status=status,
            body=body,
            headers={"X-Request": str(index)},
        )

    async def headers(
        ctx: HeaderContext,
        call_next: HeaderNext,
    ) -> HeaderDecision:
        assert ctx.headers.path is not None
        index = int(ctx.headers.path[1:])
        for ordinal, status in enumerate(cases[index][2]):
            await ctx.send(
                HTTPResponse(
                    status=status,
                    headers={
                        "X-Request": str(index),
                        "X-Interim": str(ordinal),
                    },
                )
            )
        if cases[index][3]:
            await ctx.send(response_for(index))
        return await call_next()

    def respond(ctx: ResponderContext) -> ResponseSpec:
        index = int(ctx.request.target[1:])
        faults.clear()
        cutoff = cases[index][4]
        if cutoff is not None:
            faults.append(DropConnection(cutoff))
        return response_for(index)

    harness = _harness(
        response=respond,
        header_middlewares=[headers],
        transmission=FaultyTransmission(faults),
    )
    harness.feed(
        b"".join(
            f"{case[0]} /{index} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode()
            for index, case in enumerate(cases)
        ),
        eof=True,
    )
    await harness.run()

    expected_count = _response_history_length(cases)
    assert len(harness.recorder.exchanges) == expected_count
    assert harness.responded == [
        f"/{index}" for index in range(expected_count) if not cases[index][3]
    ]
    recorded_wire = b""
    for index, exchange in enumerate(harness.recorder.exchanges):
        method, status, interim, early, cutoff, body = cases[index]
        assert exchange.request.target == f"/{index}"
        assert [r.status for r in exchange.interim_responses] == interim
        for ordinal, response in enumerate(exchange.interim_responses):
            assert response.headers["X-Request"] == str(index)
            assert response.headers["X-Interim"] == str(ordinal)
            recorded_wire += response.wire_raw_bytes
        final = exchange.response
        assert final is not None
        assert final.status == status
        assert final.headers["X-Request"] == str(index)
        wire_body = final.wire_raw_bytes.split(b"\r\n\r\n", 1)[1]
        expected_body = (
            b""
            if method == "HEAD"
            or status
            in {
                101,
                204,
                304,
            }
            else body
        )
        if not early and cutoff is not None:
            expected_body = expected_body[:cutoff]
        assert wire_body == expected_body
        recorded_wire += final.wire_raw_bytes
    assert recorded_wire == harness.written


async def _finish_close_history(
    harness: Harness,
    sleep: _HeldSleep,
    cause: str,
) -> None:
    task = asyncio.create_task(harness.connection.run())
    if cause in {"shutdown", "cancel"}:
        await asyncio.wait_for(sleep.entered.wait(), timeout=1)
        if cause == "shutdown":
            harness.connection.shutdown()
        else:
            task.cancel()
    if cause == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    else:
        await asyncio.wait_for(task, timeout=1)


def _response_history_length(
    cases: list[tuple[str, int, list[int], bool, int | None, bytes]],
) -> int:
    expected_count = len(cases)
    for index, (_, status, _, early, cutoff, _) in enumerate(cases):
        if status == 101 or (not early and cutoff is not None):
            expected_count = index + 1
            break
    return expected_count
