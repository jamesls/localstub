from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

import pytest

from localstub.http.exchange import (
    CloseReason,
    ConnectionClosed,
    RecordedExchange,
)
from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.websocket.deflate import MessageDeflater, PerMessageDeflate
from localstub.websocket.frames import FrameDecoder, Opcode, WebSocketFrame
from localstub.websocket.records import (
    WebSocketClose,
    WebSocketLog,
    WebSocketMessage,
)
from localstub.websocket.session import (
    SessionOptions,
    WebSocketEnded,
    WebSocketHandler,
    WebSocketSession,
)
from tests.unit.server.fakes import RecordingSleep, settle

MASK = b"\x0a\x0b\x0c\x0d"
TIMESTAMP = datetime(2026, 11, 24, tzinfo=UTC)
REQUEST = HTTPRequest(method="GET", target="/ws", headers=Headers.empty())
HANDSHAKE = RecordedExchange(
    request=RecordedHTTPRequest(
        request=REQUEST,
        as_received=REQUEST,
        wire_raw_bytes=b"GET /ws HTTP/1.1\r\n\r\n",
        http_version="1.1",
    ),
    response=None,
    request_timestamp=TIMESTAMP,
    response_timestamp=None,
)


class FakeWriter:
    def __init__(self, error: OSError | None = None) -> None:
        self.writes: list[bytes] = []
        self.error = error

    def write(self, data: bytes | memoryview) -> None:
        if self.error is not None:
            raise self.error
        self.writes.append(bytes(data))

    async def drain(self) -> None:
        await asyncio.sleep(0)

    @property
    def frames(self) -> list[WebSocketFrame]:
        decoded = FrameDecoder().feed(b"".join(self.writes))
        return [item.frame for item in decoded]


class FakeClock:
    def __init__(self) -> None:
        self.ticks = 0.0

    def now(self) -> float:
        self.ticks += 1.0
        return self.ticks


class RaisingSink:
    def extend(self, data: bytes | bytearray | memoryview, /) -> None:
        _ = data
        raise RuntimeError("sink failed")


@dataclass
class Harness:
    session: WebSocketSession
    reader: asyncio.StreamReader
    writer: FakeWriter
    sleep: Callable[[float], Awaitable[None]]
    closes: list[tuple[CloseReason, bool]] = field(
        default_factory=list[tuple[CloseReason, bool]]
    )
    task: asyncio.Task[None] | None = None

    def send(self, *frames: WebSocketFrame) -> None:
        for frame in frames:
            masked = frame if frame.mask_key else replace(frame, mask_key=MASK)
            self.reader.feed_data(masked.encode())

    def start(self) -> None:
        self.task = asyncio.create_task(self.session.run())

    async def finish(self) -> ConnectionClosed:
        closed = await self.session.wait_closed(timeout=1.0)
        assert self.task is not None
        await asyncio.wait_for(self.task, timeout=1.0)
        return closed

    @property
    def server_frames(self) -> list[WebSocketFrame]:
        return self.writer.frames


def _harness(
    *,
    handler: WebSocketHandler | None = None,
    options: SessionOptions | None = None,
    compression: PerMessageDeflate | None = None,
    writer: FakeWriter | None = None,
    sleep: Callable[[float], Awaitable[None]] | None = None,
    history_size: int | None = 256,
    reader: asyncio.StreamReader | None = None,
    consumed: RaisingSink | None = None,
) -> Harness:
    stream_reader = reader or asyncio.StreamReader()
    stream_writer = writer or FakeWriter()
    closes: list[tuple[CloseReason, bool]] = []
    events: list[ConnectionClosed] = []

    def close(reason: CloseReason, reset: bool) -> ConnectionClosed:
        if not events:
            closes.append((reason, reset))
            events.append(
                ConnectionClosed(
                    client=None,
                    reason=reason,
                    phase="websocket",
                    reset=reset,
                    requests_completed=1,
                    bytes_read=0,
                    bytes_consumed=0,
                    bytes_written=0,
                    timestamp=TIMESTAMP,
                )
            )
        return events[0]

    log = WebSocketLog(
        handshake=HANDSHAKE,
        subprotocol=None,
        compression=compression,
        history_size=history_size,
    )
    the_sleep = sleep or RecordingSleep()
    session = WebSocketSession(
        reader=stream_reader,
        writer=stream_writer,
        log=log,
        close=close,
        handler=handler,
        options=options or SessionOptions(),
        sleep=the_sleep,
        clock=FakeClock(),
        timestamp=lambda: TIMESTAMP,
        consumed=consumed,
    )
    return Harness(
        session=session,
        reader=stream_reader,
        writer=stream_writer,
        sleep=the_sleep,
        closes=closes,
    )


@pytest.mark.asyncio
async def test_receive_returns_client_messages_and_records_frames() -> None:
    harness = _harness()
    harness.start()
    harness.send(WebSocketFrame.text("one"), WebSocketFrame.binary(b"\x00"))

    first = await harness.session.receive(timeout=1.0)
    second = await harness.session.receive(timeout=1.0)

    assert first == WebSocketMessage("text", b"one")
    assert second == WebSocketMessage("binary", b"\x00")
    recorded = harness.session.recorded
    assert [f.frame.payload for f in recorded.client_frames] == [
        b"one",
        b"\x00",
    ]
    assert all(f.frame.mask_key == MASK for f in recorded.client_frames)
    assert [f.monotonic for f in recorded.client_frames] == [1.0, 2.0]
    assert recorded.client_frames[0].timestamp == TIMESTAMP
    assert [m.data for m in recorded.client_messages] == [b"one", b"\x00"]
    assert harness.session.request is HANDSHAKE.request
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_receive_frame_returns_control_frames_too() -> None:
    harness = _harness()
    harness.start()
    harness.send(WebSocketFrame.ping(b"hb"))

    frame = await harness.session.receive_frame(timeout=1.0)

    assert frame.frame.opcode == Opcode.PING
    assert frame.sender == "client"
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_auto_pong_answers_ping_with_same_payload() -> None:
    harness = _harness()
    harness.start()
    harness.send(WebSocketFrame.ping(b"hb"))
    await harness.session.receive_frame(timeout=1.0)
    await settle()

    assert harness.server_frames == [WebSocketFrame.pong(b"hb")]
    assert harness.session.recorded.server_frames[0].frame.opcode == (
        Opcode.PONG
    )
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_auto_pong_disabled_withholds_pong() -> None:
    harness = _harness(options=SessionOptions(auto_pong=False))
    harness.start()
    harness.send(WebSocketFrame.ping(b"a"))
    await harness.session.receive_frame(timeout=1.0)
    harness.session.auto_pong = True
    harness.send(WebSocketFrame.ping(b"b"))
    await harness.session.receive_frame(timeout=1.0)
    await settle()

    assert harness.server_frames == [WebSocketFrame.pong(b"b")]
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_client_close_is_echoed_and_ends_session() -> None:
    harness = _harness()
    harness.start()
    harness.send(WebSocketFrame.text("last"), WebSocketFrame.close(4001, "x"))

    closed = await harness.finish()

    assert closed.reason == "websocket_close"
    assert harness.server_frames == [WebSocketFrame(Opcode.CLOSE, b"\x0f\xa1")]
    recorded = harness.session.recorded
    assert recorded.client_close == WebSocketClose(4001, "x")
    assert recorded.server_close == WebSocketClose(4001)
    assert recorded.ended
    assert recorded.closed is closed
    assert await harness.session.receive() == WebSocketMessage("text", b"last")
    with pytest.raises(WebSocketEnded):
        await harness.session.receive()


@pytest.mark.asyncio
async def test_empty_client_close_is_echoed_empty() -> None:
    harness = _harness()
    harness.start()
    harness.send(WebSocketFrame.close(None))

    await harness.finish()

    assert harness.server_frames == [WebSocketFrame(Opcode.CLOSE, b"")]
    assert harness.session.recorded.client_close == WebSocketClose(None)


@pytest.mark.asyncio
async def test_auto_close_disabled_records_close_and_waits_for_eof() -> None:
    harness = _harness(options=SessionOptions(auto_close=False))
    harness.start()
    harness.send(WebSocketFrame.close(1000))
    await harness.session.receive_frame(timeout=1.0)
    await settle()
    assert not harness.session.ended
    harness.reader.feed_eof()

    closed = await harness.finish()

    assert closed.reason == "client"
    assert harness.server_frames == []
    assert harness.session.recorded.client_close == WebSocketClose(1000)


@pytest.mark.asyncio
async def test_close_waits_for_client_close_then_ends() -> None:
    harness = _harness(sleep=asyncio.sleep)
    harness.start()

    closing = asyncio.create_task(harness.session.close(1001, "away"))
    await settle()
    assert not harness.session.ended
    harness.send(WebSocketFrame.close(1001))
    await asyncio.wait_for(closing, timeout=1.0)
    closed = await harness.finish()

    assert closed.reason == "websocket_close"
    assert harness.server_frames == [WebSocketFrame.close(1001, "away")]
    assert harness.session.recorded.server_close == WebSocketClose(
        1001, "away"
    )
    assert harness.session.recorded.client_close == WebSocketClose(1001)


@pytest.mark.asyncio
async def test_close_times_out_on_injected_sleep() -> None:
    sleep = RecordingSleep()
    harness = _harness(sleep=sleep, options=SessionOptions(close_timeout=2.5))
    harness.start()

    await harness.session.close()
    closed = await harness.finish()

    assert closed.reason == "websocket_close"
    assert sleep.calls == [2.5]
    assert harness.session.recorded.client_close is None


@pytest.mark.asyncio
async def test_close_sends_codes_as_given_and_none_as_empty() -> None:
    harness = _harness()
    harness.start()

    await harness.session.close(None)
    await harness.session.close(1000)
    await harness.finish()

    assert harness.server_frames == [WebSocketFrame(Opcode.CLOSE, b"")]


@pytest.mark.asyncio
async def test_eof_after_server_close_counts_as_closing_handshake() -> None:
    harness = _harness(sleep=asyncio.sleep)
    harness.start()

    closing = asyncio.create_task(harness.session.close(1000))
    await settle()
    harness.reader.feed_eof()
    await asyncio.wait_for(closing, timeout=1.0)

    assert (await harness.finish()).reason == "websocket_close"


@pytest.mark.asyncio
async def test_handler_return_closes_with_1000() -> None:
    async def handler(ws: WebSocketSession) -> None:
        message = await ws.receive()
        await ws.send(message)

    harness = _harness(handler=handler)
    harness.start()
    harness.send(WebSocketFrame.text("echo"))

    closed = await harness.finish()

    assert closed.reason == "websocket_close"
    assert harness.server_frames == [
        WebSocketFrame.text("echo"),
        WebSocketFrame.close(1000),
    ]
    [sent] = harness.session.recorded.server_messages
    assert sent.message == WebSocketMessage("text", b"echo")


@pytest.mark.asyncio
async def test_handler_raise_closes_with_1011_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(ws: WebSocketSession) -> None:
        _ = ws
        raise RuntimeError("handler bug")

    harness = _harness(handler=handler)
    harness.start()

    with caplog.at_level(logging.ERROR):
        closed = await harness.finish()

    assert closed.reason == "error"
    assert harness.server_frames == [WebSocketFrame.close(1011)]
    assert "handler bug" in caplog.text


@pytest.mark.asyncio
async def test_handler_that_outlives_session_is_cancelled() -> None:
    cancelled = asyncio.Event()

    async def handler(ws: WebSocketSession) -> None:
        _ = ws
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    sleep = RecordingSleep()
    harness = _harness(handler=handler, sleep=sleep)
    harness.start()
    await settle()

    await harness.session.drop()
    await harness.finish()

    assert cancelled.is_set()
    assert sleep.calls == [5.0]


@pytest.mark.asyncio
async def test_handler_owns_its_task() -> None:
    seen: list[bool] = []

    async def handler(ws: WebSocketSession) -> None:
        seen.append(ws.owns_current_task())

    harness = _harness(handler=handler)
    assert not harness.session.owns_current_task()
    harness.start()
    await harness.finish()

    assert seen == [True]
    assert not harness.session.owns_current_task()


@pytest.mark.asyncio
async def test_strict_violation_closes_with_its_code() -> None:
    harness = _harness()
    harness.start()
    harness.reader.feed_data(WebSocketFrame.text("unmasked").encode())

    closed = await harness.finish()

    assert closed.reason == "protocol_error"
    assert harness.server_frames == [WebSocketFrame.close(1002)]
    [violation] = harness.session.recorded.violations
    assert violation.kind == "unmasked_frame"


@pytest.mark.asyncio
async def test_strict_invalid_close_is_recorded_and_fails() -> None:
    harness = _harness()
    harness.start()
    harness.send(WebSocketFrame.close(1005))

    closed = await harness.finish()

    assert closed.reason == "protocol_error"
    assert harness.session.recorded.client_close == WebSocketClose(1005)
    assert harness.server_frames == [WebSocketFrame.close(1002)]


@pytest.mark.asyncio
async def test_lenient_violation_is_recorded_and_reading_continues() -> None:
    harness = _harness(options=SessionOptions(strict=False))
    harness.start()
    harness.reader.feed_data(WebSocketFrame(Opcode.TEXT, b"\xff").encode())

    message = await harness.session.receive(timeout=1.0)

    assert message.data == b"\xff"
    kinds = [v.kind for v in harness.session.recorded.violations]
    assert kinds == ["unmasked_frame", "invalid_utf8"]
    assert not harness.session.ended
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_strict_oversized_frame_fails_before_payload_arrives() -> None:
    harness = _harness(options=SessionOptions(max_message_bytes=100))
    harness.start()
    harness.reader.feed_data(b"\x82\xfe\x10\x00" + MASK + b"partial")

    closed = await harness.finish()

    assert closed.reason == "protocol_error"
    assert harness.server_frames == [WebSocketFrame.close(1009)]
    [violation] = harness.session.recorded.violations
    assert violation.kind == "message_too_big"
    assert harness.session.recorded.client_frames == []


@pytest.mark.asyncio
async def test_receive_queue_evicts_oldest_and_counts() -> None:
    harness = _harness(options=SessionOptions(queue_size=2))
    harness.start()
    harness.send(*(WebSocketFrame.text(str(i)) for i in range(3)))
    await settle()

    received = [await harness.session.receive(timeout=1.0) for _ in range(2)]

    assert [m.text for m in received] == ["1", "2"]
    assert harness.session.dropped_unread_messages == 1
    assert harness.session.dropped_unread_frames == 1
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_receive_queue_evicts_oldest_past_byte_bound() -> None:
    options = SessionOptions(queue_size=None, queue_bytes=5)
    harness = _harness(options=options)
    harness.start()
    harness.send(*(WebSocketFrame.text(t) for t in ("aaa", "bb", "ccc")))
    await settle()

    received = [await harness.session.receive(timeout=1.0) for _ in range(2)]

    assert [m.text for m in received] == ["bb", "ccc"]
    assert harness.session.dropped_unread_messages == 1
    # Each frame record holds more than 5 bytes, so only the newest stays.
    assert harness.session.dropped_unread_frames == 2
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_record_histories_are_bounded() -> None:
    harness = _harness(history_size=2)
    harness.start()
    harness.send(*(WebSocketFrame.text(str(i)) for i in range(3)))
    await settle()

    recorded = harness.session.recorded
    assert [f.frame.payload for f in recorded.frames] == [b"1", b"2"]
    assert recorded.dropped_frames == 1
    assert recorded.dropped_messages == 1
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_receive_times_out() -> None:
    harness = _harness()
    harness.start()

    with pytest.raises(TimeoutError):
        await harness.session.receive(timeout=0.01)
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_async_iteration_ends_with_session() -> None:
    harness = _harness()
    harness.start()
    harness.send(WebSocketFrame.text("a"), WebSocketFrame.close(1000))

    messages = [message.text async for message in harness.session]

    assert messages == ["a"]
    await harness.finish()


@pytest.mark.asyncio
async def test_concurrent_fragmented_sends_do_not_interleave() -> None:
    harness = _harness()
    harness.start()

    await asyncio.gather(
        harness.session.send_text("aaaa", fragment_size=1),
        harness.session.send_bytes(b"bbbb", fragment_size=2),
    )

    frames = harness.server_frames
    assert frames == [
        WebSocketFrame.text("a", fin=False),
        WebSocketFrame.continuation(b"a", fin=False),
        WebSocketFrame.continuation(b"a", fin=False),
        WebSocketFrame.continuation(b"a"),
        WebSocketFrame.binary(b"bb", fin=False),
        WebSocketFrame.continuation(b"bb"),
    ]
    messages = harness.session.recorded.server_messages
    assert [len(m.frames) for m in messages] == [4, 2]
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_auto_pong_may_land_between_fragments() -> None:
    harness = _harness()
    harness.start()
    sending = asyncio.create_task(
        harness.session.send_text("abc", fragment_size=1)
    )
    harness.send(WebSocketFrame.ping(b"p"))
    await sending
    await settle()

    opcodes = [frame.opcode for frame in harness.server_frames]
    assert sorted(opcodes) == [0, 0, 1, Opcode.PONG]
    assert opcodes[0] == Opcode.TEXT
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_write_chunk_size_splits_frame_writes() -> None:
    harness = _harness(options=SessionOptions(write_chunk_size=3))
    harness.start()

    await harness.session.send_bytes(b"12345678")

    assert harness.writer.writes == [b"\x82\x08\x31", b"234", b"567", b"8"]
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_compressed_session_compresses_unless_told_not_to() -> None:
    harness = _harness(compression=PerMessageDeflate())
    harness.start()
    deflater = MessageDeflater(no_context_takeover=False, max_window_bits=None)

    await harness.session.send_text("Hello")
    await harness.session.send_text("Hello", compress=False)
    harness.send(
        WebSocketFrame(Opcode.TEXT, deflater.compress(b"Hi"), rsv1=True)
    )
    received = await harness.session.receive(timeout=1.0)

    first, second = harness.server_frames
    assert first == WebSocketFrame(
        Opcode.TEXT, bytes.fromhex("f248cdc9c90700"), rsv1=True
    )
    assert second == WebSocketFrame.text("Hello")
    assert received == WebSocketMessage("text", b"Hi")
    sent = harness.session.recorded.server_messages
    assert [m.compressed for m in sent] == [True, False]
    assert [m.data for m in sent] == [b"Hello", b"Hello"]
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_compressed_fragments_set_rsv1_on_first_frame_only() -> None:
    harness = _harness(compression=PerMessageDeflate())
    harness.start()

    await harness.session.send_bytes(bytes(10), fragment_size=2)

    frames = harness.server_frames
    assert len(frames) > 1
    assert [frame.rsv1 for frame in frames] == [True] + [False] * (
        len(frames) - 1
    )
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_send_argument_errors() -> None:
    harness = _harness()

    with pytest.raises(ValueError, match="not negotiated"):
        await harness.session.send_text("x", compress=True)
    with pytest.raises(ValueError, match="fragment_size"):
        await harness.session.send_text("x", fragment_size=0)
    with pytest.raises(ValueError, match="125 bytes"):
        await harness.session.ping(bytes(126))


@pytest.mark.asyncio
async def test_ping_and_pong_send_control_frames() -> None:
    harness = _harness()
    harness.start()

    await harness.session.ping(b"1")
    await harness.session.pong(b"2")

    assert harness.server_frames == [
        WebSocketFrame.ping(b"1"),
        WebSocketFrame.pong(b"2"),
    ]
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_send_frame_close_records_but_keeps_session_open() -> None:
    harness = _harness()
    harness.start()

    await harness.session.send_frame(WebSocketFrame.close(1000))
    await harness.session.send_text("after close")
    await harness.session.send_frame(WebSocketFrame(0x3, b"", rsv2=True))

    assert harness.session.recorded.server_close == WebSocketClose(1000)
    assert not harness.session.ended
    assert [f.opcode for f in harness.server_frames] == [
        Opcode.CLOSE,
        Opcode.TEXT,
        0x3,
    ]
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_send_raw_writes_bytes_without_recording_a_frame() -> None:
    harness = _harness()
    harness.start()
    frame = WebSocketFrame.text("partial update")

    await harness.session.send_raw(frame.encode()[:5])
    await harness.session.drop(reset=True)
    closed = await harness.finish()

    assert harness.writer.writes == [frame.encode()[:5]]
    assert harness.session.recorded.server_frames == []
    assert closed.reason == "websocket_drop"
    assert closed.reset
    assert harness.closes == [("websocket_drop", True)]


@pytest.mark.asyncio
async def test_send_after_end_raises() -> None:
    harness = _harness()
    harness.start()
    await harness.session.drop()
    await harness.finish()

    with pytest.raises(WebSocketEnded):
        await harness.session.send_text("late")
    with pytest.raises(WebSocketEnded):
        await harness.session.send_raw(b"late")
    with pytest.raises(WebSocketEnded):
        await harness.session.receive_frame()
    await harness.session.drop()
    await harness.session.close()
    assert harness.closes == [("websocket_drop", False)]


@pytest.mark.parametrize(
    ("error", "reset"),
    [(ConnectionResetError(), True), (BrokenPipeError(), False)],
)
@pytest.mark.asyncio
async def test_write_failure_ends_with_client(
    error: OSError, reset: bool
) -> None:
    harness = _harness(writer=FakeWriter(error))
    harness.start()

    with pytest.raises(WebSocketEnded):
        await harness.session.send_text("x")
    closed = await harness.finish()

    assert closed.reason == "client"
    assert closed.reset == reset


@pytest.mark.asyncio
async def test_close_after_lost_write_returns_quietly() -> None:
    harness = _harness(writer=FakeWriter(BrokenPipeError()))
    harness.start()

    await harness.session.close()

    assert (await harness.finish()).reason == "client"


@pytest.mark.asyncio
async def test_read_reset_ends_with_client_reset() -> None:
    harness = _harness()
    harness.start()
    harness.reader.set_exception(ConnectionResetError())

    closed = await harness.finish()

    assert closed.reason == "client"
    assert closed.reset


@pytest.mark.asyncio
async def test_reader_loop_error_ends_with_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    harness = _harness(consumed=RaisingSink())
    harness.start()
    harness.send(WebSocketFrame.text("x"))

    with caplog.at_level(logging.ERROR):
        closed = await harness.finish()

    assert closed.reason == "error"
    assert "sink failed" in caplog.text


@pytest.mark.asyncio
async def test_cancelled_run_ends_with_shutdown() -> None:
    harness = _harness()
    harness.start()
    await settle()
    assert harness.task is not None

    harness.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await harness.task

    assert harness.closes == [("shutdown", False)]
    assert harness.session.ended


@pytest.mark.asyncio
async def test_terminate_ends_session_with_given_event() -> None:
    harness = _harness()
    harness.start()
    event = ConnectionClosed(
        client=None,
        reason="shutdown",
        phase="websocket",
        reset=False,
        requests_completed=1,
        bytes_read=0,
        bytes_consumed=0,
        bytes_written=0,
        timestamp=TIMESTAMP,
    )

    harness.session.terminate(event)
    closed = await harness.finish()

    assert closed is event
    assert harness.closes == []
    with pytest.raises(WebSocketEnded):
        await harness.session.receive()


@pytest.mark.asyncio
async def test_close_after_unanswered_client_close_ends_at_once() -> None:
    sleep = RecordingSleep()
    harness = _harness(options=SessionOptions(auto_close=False), sleep=sleep)
    harness.start()
    harness.send(WebSocketFrame.close(1000))
    await harness.session.receive_frame(timeout=1.0)

    await harness.session.close(1000)
    closed = await harness.finish()

    assert closed.reason == "websocket_close"
    assert sleep.calls == []
    assert harness.server_frames == [WebSocketFrame.close(1000)]


@pytest.mark.asyncio
async def test_handler_stopped_by_session_end_sends_nothing() -> None:
    async def handler(ws: WebSocketSession) -> None:
        await ws.drop()
        await ws.send_text("too late")

    harness = _harness(handler=handler)
    harness.start()

    closed = await harness.finish()

    assert closed.reason == "websocket_drop"
    assert harness.server_frames == []


@pytest.mark.asyncio
async def test_strict_violation_with_lost_writer_ends_with_client() -> None:
    harness = _harness(writer=FakeWriter(BrokenPipeError()))
    harness.start()
    harness.reader.feed_data(WebSocketFrame.text("unmasked").encode())

    closed = await harness.finish()

    assert closed.reason == "client"
    assert [v.kind for v in harness.session.recorded.violations] == [
        "unmasked_frame"
    ]


@pytest.mark.asyncio
async def test_pong_to_lost_writer_ends_with_client() -> None:
    harness = _harness(writer=FakeWriter(ConnectionResetError()))
    harness.start()
    harness.send(WebSocketFrame.ping())

    closed = await harness.finish()

    assert closed.reason == "client"
    assert closed.reset


@pytest.mark.asyncio
async def test_partial_frame_within_limits_waits_for_rest() -> None:
    harness = _harness(options=SessionOptions(max_message_bytes=10))
    harness.start()
    wire = replace(WebSocketFrame.text("short"), mask_key=MASK).encode()
    harness.reader.feed_data(wire[:7])
    await settle()
    harness.reader.feed_data(wire[7:])

    assert await harness.session.receive(timeout=1.0) == WebSocketMessage(
        "text", b"short"
    )
    await harness.session.drop()
    await harness.finish()


@pytest.mark.asyncio
async def test_terminate_after_end_keeps_first_event() -> None:
    harness = _harness()
    harness.start()
    await harness.session.drop()
    first = await harness.finish()

    harness.session.terminate(replace(first, reason="shutdown"))

    assert harness.session.recorded.closed is first


async def _never(seconds: float) -> None:
    _ = seconds
    await asyncio.Event().wait()


async def _echo(ws: WebSocketSession) -> None:
    async for message in ws:
        await ws.send(message)


@pytest.mark.asyncio
async def test_handler_answers_queued_messages_before_violation() -> None:
    harness = _harness(handler=_echo, sleep=_never)
    harness.start()
    harness.send(WebSocketFrame.text("one"), WebSocketFrame.text("two"))
    harness.reader.feed_data(WebSocketFrame.text("unmasked").encode())

    closed = await harness.finish()

    assert closed.reason == "protocol_error"
    assert harness.server_frames == [
        WebSocketFrame.text("one"),
        WebSocketFrame.text("two"),
        WebSocketFrame.close(1002),
    ]


@pytest.mark.asyncio
async def test_handler_answers_queued_messages_before_close_echo() -> None:
    harness = _harness(handler=_echo, sleep=_never)
    harness.start()
    harness.send(
        WebSocketFrame.text("one"),
        WebSocketFrame.text("two"),
        WebSocketFrame.close(1000),
    )

    closed = await harness.finish()

    assert closed.reason == "websocket_close"
    assert harness.server_frames == [
        WebSocketFrame.text("one"),
        WebSocketFrame.text("two"),
        WebSocketFrame.close(1000),
    ]


@pytest.mark.asyncio
async def test_handler_closing_with_queued_messages_stops_the_wait() -> None:
    async def handler(ws: WebSocketSession) -> None:
        await ws.receive()

    harness = _harness(handler=handler, sleep=_never)
    harness.start()
    harness.send(
        WebSocketFrame.text("one"),
        WebSocketFrame.text("two"),
        WebSocketFrame.text("three"),
    )
    harness.reader.feed_data(WebSocketFrame.text("unmasked").encode())

    # The handler's close() waits on the reader for the client's Close,
    # so the reader must not keep waiting on the handler.
    closed = await harness.finish()

    assert closed.reason == "protocol_error"
    assert harness.server_frames[0] == WebSocketFrame.close(1000)


@pytest.mark.asyncio
async def test_stalled_handler_delays_violation_close_by_close_timeout() -> (
    None
):
    async def handler(ws: WebSocketSession) -> None:
        _ = ws
        await asyncio.Event().wait()

    sleep = RecordingSleep()
    options = SessionOptions(close_timeout=2.5)
    harness = _harness(handler=handler, sleep=sleep, options=options)
    harness.start()
    harness.send(WebSocketFrame.text("one"))
    harness.reader.feed_data(WebSocketFrame.text("unmasked").encode())

    closed = await harness.finish()

    assert closed.reason == "protocol_error"
    assert harness.server_frames == [WebSocketFrame.close(1002)]
    # The catch-up wait, then the wait for the handler to finish.
    assert sleep.calls == [2.5, 2.5]


async def _close_in_background(ws: WebSocketSession) -> None:
    asyncio.create_task(ws.close(1000))
    await settle()


async def _send_close_frame(ws: WebSocketSession) -> None:
    await ws.send_frame(WebSocketFrame.close(1000))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "send_close",
    [_close_in_background, _send_close_frame],
    ids=["close", "send_frame"],
)
async def test_violation_after_server_close_sends_no_second_close(
    send_close: Callable[[WebSocketSession], Awaitable[None]],
) -> None:
    harness = _harness(sleep=_never)
    harness.start()
    await send_close(harness.session)

    harness.reader.feed_data(WebSocketFrame.text("unmasked").encode())
    closed = await harness.finish()

    assert closed.reason == "protocol_error"
    assert harness.server_frames == [WebSocketFrame.close(1000)]


@pytest.mark.asyncio
async def test_client_close_after_sent_close_frame_is_not_echoed() -> None:
    harness = _harness()
    harness.start()
    await harness.session.send_frame(WebSocketFrame.close(4000))

    harness.send(WebSocketFrame.close(1000))
    closed = await harness.finish()

    assert closed.reason == "websocket_close"
    assert harness.server_frames == [WebSocketFrame.close(4000)]
