"""One server-side WebSocket session: the reader loop and the controls.

``WebSocketSession`` is what a test or a handler drives once the
opening handshake is on the wire.  Its reader loop records every
client frame whether or not anything awaits ``receive()``, answers
pings and Close frames as the flags say, and fails the connection on a
protocol violation in strict mode.

The session never imports the server or the recorder.  It is built
from its collaborators: a byte stream, a writer, the log behind its
record, and a close callback, so the connection stays the only place
a close is decided.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from localstub.clock import Clock
from localstub.http import stream
from localstub.http.exchange import CloseReason, ConnectionClosed
from localstub.http.request import RecordedHTTPRequest
from localstub.websocket.assembly import MAX_CONTROL_PAYLOAD, MessageAssembler
from localstub.websocket.deflate import MessageDeflater
from localstub.websocket.frames import (
    DecodedFrame,
    FrameDecoder,
    Opcode,
    WebSocketFrame,
    close_payload,
)
from localstub.websocket.records import (
    BoundedHistory,
    RecordedWebSocket,
    RecordedWebSocketFrame,
    RecordedWebSocketMessage,
    Sender,
    WebSocketClose,
    WebSocketLog,
    WebSocketMessage,
    WebSocketViolation,
    frame_size,
)

LOG = logging.getLogger(__name__)

DEFAULT_MAX_READ = 64 * 1024

type WebSocketHandler = Callable[[WebSocketSession], Awaitable[None]]
type Sleep = Callable[[float], Awaitable[None]]
type CloseSession = Callable[[CloseReason, bool], ConnectionClosed]
"""Decide the connection's close with a reason and a reset flag, start
closing the transport, and return the event (the earlier one, when a
close was already decided)."""


class WebSocketEnded(Exception):
    """The session has ended; nothing more can be received or sent."""


class SessionWriter(Protocol):
    def write(self, data: bytes | memoryview) -> None: ...

    async def drain(self) -> None: ...


class ByteSink(Protocol):
    def extend(self, data: bytes | bytearray | memoryview, /) -> None: ...


@dataclass(frozen=True)
class SessionOptions:
    """How a session behaves; the server builds it from the accept."""

    strict: bool = True
    auto_pong: bool = True
    auto_close: bool = True
    close_timeout: float = 5.0
    max_message_bytes: int | None = None
    queue_size: int | None = 256
    """Entry bound of the ``receive()`` and ``receive_frame()`` queues."""
    queue_bytes: int | None = None
    """Byte bound of the same queues; ``None`` is none."""
    write_chunk_size: int | None = None
    max_read: int = DEFAULT_MAX_READ


class _Inbox[T]:
    """Evict-oldest queue whose readers learn when the session ends.

    Entries queued before the end are still handed out; once the queue
    is empty and closed, ``get()`` raises ``WebSocketEnded``.
    ``wait_drained()`` returns once a reader is waiting on an empty
    queue, which is how the session knows a handler has dealt with
    every message it was given.
    """

    def __init__(
        self,
        maxsize: int | None,
        max_bytes: int | None,
        size: Callable[[T], int],
    ) -> None:
        self._items: BoundedHistory[T] = BoundedHistory(
            maxsize, max_bytes, size
        )
        self._changed = asyncio.Event()
        self._drained = asyncio.Event()
        self._waiting = 0
        self._closed = False

    @property
    def dropped(self) -> int:
        return self._items.dropped

    @property
    def pending(self) -> bool:
        return bool(len(self._items))

    def put(self, item: T) -> None:
        self._items.append(item)
        self._drained.clear()
        self._changed.set()

    def close(self) -> None:
        self._closed = True
        self._changed.set()

    async def get(self) -> T:
        while not len(self._items):
            if self._closed:
                raise WebSocketEnded("the WebSocket session has ended")
            self._changed.clear()
            self._waiting += 1
            self._drained.set()
            try:
                await self._changed.wait()
            finally:
                self._waiting -= 1
        return self._items.popleft()

    async def wait_drained(self) -> None:
        while len(self._items) or not self._waiting:
            self._drained.clear()
            await self._drained.wait()


async def _cancel(*tasks: asyncio.Task[object] | None) -> None:
    pending = [task for task in tasks if task is not None]
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


class WebSocketSession:
    """A WebSocket session a test or a handler drives.

    ``receive()`` and the send methods raise ``WebSocketEnded`` once
    the session has ended, and a timeout raises ``TimeoutError``.
    Messages the client sent before the end are still returned by
    ``receive()`` before it raises.  ``auto_pong`` and ``auto_close``
    start from the accept's values and may be changed at any time.
    """

    def __init__(
        self,
        *,
        reader: stream.ByteStream,
        writer: SessionWriter,
        log: WebSocketLog,
        close: CloseSession,
        handler: WebSocketHandler | None,
        options: SessionOptions,
        sleep: Sleep,
        clock: Clock,
        timestamp: Callable[[], datetime],
        consumed: ByteSink | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._log = log
        self._record = RecordedWebSocket(log)
        self._close = close
        self._handler = handler
        self._options = options
        self._sleep = sleep
        self._clock = clock
        self._timestamp = timestamp
        self._consumed = consumed
        self.auto_pong = options.auto_pong
        self.auto_close = options.auto_close
        compression = log.compression
        self._deflater: MessageDeflater | None = None
        if compression is not None:
            self._deflater = MessageDeflater(
                no_context_takeover=compression.server_no_context_takeover,
                max_window_bits=compression.server_max_window_bits,
            )
        self._assembler = MessageAssembler(
            sender="client",
            compression=compression,
            strict=options.strict,
            max_message_bytes=options.max_message_bytes,
        )
        self._messages: _Inbox[WebSocketMessage] = _Inbox(
            options.queue_size, options.queue_bytes, _data_size
        )
        self._frames: _Inbox[RecordedWebSocketFrame] = _Inbox(
            options.queue_size, options.queue_bytes, frame_size
        )
        self._frame_lock = asyncio.Lock()
        self._message_lock = asyncio.Lock()
        self._client_closed = asyncio.Event()
        self._closing = asyncio.Event()
        self._ended = False
        self._handler_task: asyncio.Task[None] | None = None

    @property
    def request(self) -> RecordedHTTPRequest:
        """The handshake request."""
        return self._log.handshake.request

    @property
    def recorded(self) -> RecordedWebSocket:
        """The session's record: frames, messages, closes, violations."""
        return self._record

    @property
    def ended(self) -> bool:
        return self._ended

    @property
    def dropped_unread_messages(self) -> int:
        """Messages evicted unread from the ``receive()`` queue."""
        return self._messages.dropped

    @property
    def dropped_unread_frames(self) -> int:
        """Frames evicted unread from the ``receive_frame()`` queue."""
        return self._frames.dropped

    def owns_current_task(self) -> bool:
        """Whether the calling code runs in this session's handler."""
        task = self._handler_task
        return task is not None and task is asyncio.current_task()

    # -- receiving -----------------------------------------------------

    async def receive(self, timeout: float | None = None) -> WebSocketMessage:
        """Return the next data message from the client."""
        return await asyncio.wait_for(self._messages.get(), timeout)

    async def receive_frame(
        self, timeout: float | None = None
    ) -> RecordedWebSocketFrame:
        """Return the next client frame of any opcode."""
        return await asyncio.wait_for(self._frames.get(), timeout)

    def __aiter__(self) -> WebSocketSession:
        return self

    async def __anext__(self) -> WebSocketMessage:
        try:
            return await self._messages.get()
        except WebSocketEnded:
            raise StopAsyncIteration from None

    async def wait_closed(
        self, timeout: float | None = None
    ) -> ConnectionClosed:
        """Wait for the session to end and return its close event."""
        await self._record.wait_ended(timeout)
        closed = self._log.closed
        assert closed is not None
        return closed

    # -- sending -------------------------------------------------------

    async def send_text(
        self,
        text: str,
        *,
        fragment_size: int | None = None,
        compress: bool | None = None,
    ) -> None:
        """Send a text message, in frames of ``fragment_size`` bytes."""
        await self.send(
            WebSocketMessage("text", text.encode("utf-8")),
            fragment_size=fragment_size,
            compress=compress,
        )

    async def send_bytes(
        self,
        data: bytes,
        *,
        fragment_size: int | None = None,
        compress: bool | None = None,
    ) -> None:
        """Send a binary message, in frames of ``fragment_size`` bytes."""
        await self.send(
            WebSocketMessage("binary", data),
            fragment_size=fragment_size,
            compress=compress,
        )

    async def send(
        self,
        message: WebSocketMessage,
        *,
        fragment_size: int | None = None,
        compress: bool | None = None,
    ) -> None:
        """Send ``message`` with its kind, as an echo does.

        ``compress=None`` compresses when permessage-deflate was
        negotiated; ``False`` sends the message uncompressed, and
        ``True`` on a session without the extension raises
        ``ValueError``.
        """
        if fragment_size is not None and fragment_size < 1:
            raise ValueError("fragment_size must be at least 1 or None")
        self._check_open()
        deflater = self._deflater
        if compress and deflater is None:
            raise ValueError("permessage-deflate was not negotiated")
        use_deflate = deflater is not None and compress is not False
        opcode = Opcode.TEXT if message.kind == "text" else Opcode.BINARY
        async with self._message_lock:
            self._check_open()
            payload = message.data
            if use_deflate:
                assert deflater is not None
                payload = deflater.compress(payload)
            frames = _fragments(opcode, payload, fragment_size, use_deflate)
            sent = [await self._send_frame(frame) for frame in frames]
            self._log.add_message(
                RecordedWebSocketMessage(
                    message=message,
                    sender="server",
                    frames=tuple(sent),
                    compressed=use_deflate,
                    timestamp=sent[-1].timestamp,
                )
            )

    async def ping(self, payload: bytes = b"") -> None:
        await self._send_control(Opcode.PING, payload)

    async def pong(self, payload: bytes = b"") -> None:
        await self._send_control(Opcode.PONG, payload)

    async def close(self, code: int | None = 1000, reason: str = "") -> None:
        """Start the closing handshake and end the session.

        Sends a Close frame with any 16-bit ``code`` as given, waits up
        to ``close_timeout`` for the client's Close, then closes the
        transport.  ``code=None`` sends an empty Close payload.  Does
        nothing once the session has ended.
        """
        payload = close_payload(code, reason)
        if self._ended:
            return
        self._closing.set()
        try:
            await self._send_frame(WebSocketFrame(Opcode.CLOSE, payload))
        except WebSocketEnded:
            return
        if not self._client_closed.is_set():
            await self._wait_for_client_close()
        self._end("websocket_close")

    # -- faults --------------------------------------------------------

    async def send_frame(self, frame: WebSocketFrame) -> None:
        """Write ``frame`` exactly as given, valid or not, and record it.

        A Close frame is recorded as ``server_close`` but does not start
        the closing handshake.
        """
        await self._send_frame(frame)

    async def send_raw(self, data: bytes) -> None:
        """Write ``data`` with no framing; it is not recorded as a frame."""
        self._check_open()
        async with self._frame_lock:
            self._check_open()
            await self._write(data)

    async def drop(self, *, reset: bool = False) -> None:
        """Close the transport with no Close frame.

        A client reports this as closure code 1006.  ``reset=True``
        requests an abortive TCP close.  Does nothing once the session
        has ended.
        """
        self._end("websocket_drop", reset=reset)

    # -- lifecycle -----------------------------------------------------

    async def run(self) -> None:
        """Run the reader loop and the handler until the session ends.

        A handler still running ``close_timeout`` seconds after the
        session ended is cancelled.  Cancelling ``run()`` ends the
        session with ``shutdown`` and cancels both.
        """
        reader = asyncio.create_task(self._read_loop())
        handler = self._handler
        if handler is not None:
            self._handler_task = asyncio.create_task(
                self._run_handler(handler)
            )
        try:
            await self._log.ended.wait()
            await _cancel(reader)
            await self._settle_handler()
        except asyncio.CancelledError:
            self._end("shutdown")
            await _cancel(reader, self._handler_task)
            raise

    def terminate(self, closed: ConnectionClosed) -> None:
        """End the session with a close decided outside it."""
        self._finish(closed)

    async def _settle_handler(self) -> None:
        task = self._handler_task
        if task is None or task.done():
            return
        timer = asyncio.ensure_future(self._sleep(self._options.close_timeout))
        try:
            await asyncio.wait({task, timer}, return_when="FIRST_COMPLETED")
        finally:
            await _cancel(task, timer)

    async def _run_handler(self, handler: WebSocketHandler) -> None:
        try:
            await handler(self)
        except WebSocketEnded:
            return
        except Exception:
            LOG.exception("Error in WebSocket handler")
            await self._fail(1011, "error")
            return
        await self.close(1000)

    def _check_open(self) -> None:
        if self._ended:
            raise WebSocketEnded("the WebSocket session has ended")

    def _end(self, reason: CloseReason, *, reset: bool = False) -> None:
        if self._ended:
            return
        self._finish(self._close(reason, reset))

    def _finish(self, closed: ConnectionClosed) -> None:
        if self._ended:
            return
        self._ended = True
        self._log.end(closed)
        self._messages.close()
        self._frames.close()
        # An ended session can stay in the server's history for a long
        # time; its compression contexts are a few hundred KiB each.
        self._deflater = None
        self._assembler.release()

    async def _let_handler_catch_up(self) -> None:
        """Wait for a handler to deal with the messages it was given.

        The reader loop runs ahead of the handler, so a violation or a
        Close can arrive while earlier messages are still queued.  A
        server that handled frames in order would have answered those
        messages first, so the session waits, up to ``close_timeout``,
        until the handler asks for a message the queue does not have,
        returns, or starts closing; a closing handler answers nothing
        more and waits on this reader for the client's Close.
        """
        task = self._handler_task
        if (
            task is None
            or task.done()
            or self._closing.is_set()
            or not self._messages.pending
        ):
            return
        waiters = (
            asyncio.ensure_future(self._messages.wait_drained()),
            asyncio.ensure_future(self._closing.wait()),
            asyncio.ensure_future(self._sleep(self._options.close_timeout)),
        )
        try:
            await asyncio.wait({task, *waiters}, return_when="FIRST_COMPLETED")
        finally:
            await _cancel(*waiters)

    async def _fail(self, code: int, reason: CloseReason) -> None:
        """Send a Close frame with ``code`` and end with ``reason``."""
        if reason == "protocol_error":
            await self._let_handler_catch_up()
        try:
            await self._send_frame(WebSocketFrame.close(code))
        except WebSocketEnded:
            return
        self._end(reason)

    async def _wait_for_client_close(self) -> None:
        waiters = (
            asyncio.create_task(self._client_closed.wait()),
            asyncio.create_task(self._log.ended.wait()),
            asyncio.ensure_future(self._sleep(self._options.close_timeout)),
        )
        try:
            await asyncio.wait(waiters, return_when="FIRST_COMPLETED")
        finally:
            for waiter in waiters:
                waiter.cancel()
            await asyncio.gather(*waiters, return_exceptions=True)

    # -- wire ----------------------------------------------------------

    def _record_frame(
        self,
        frame: WebSocketFrame,
        sender: Sender,
        wire: bytes,
    ) -> RecordedWebSocketFrame:
        recorded = RecordedWebSocketFrame(
            frame=frame,
            sender=sender,
            wire_raw_bytes=wire,
            timestamp=self._timestamp(),
            monotonic=self._clock.now(),
        )
        self._log.add_frame(recorded)
        return recorded

    async def _send_control(self, opcode: Opcode, payload: bytes) -> None:
        if len(payload) > MAX_CONTROL_PAYLOAD:
            raise ValueError("control frame payload is limited to 125 bytes")
        await self._send_frame(WebSocketFrame(opcode, payload))

    async def _send_frame(
        self, frame: WebSocketFrame
    ) -> RecordedWebSocketFrame:
        self._check_open()
        wire = frame.encode()
        async with self._frame_lock:
            self._check_open()
            recorded = self._record_frame(frame, "server", wire)
            if frame.opcode == Opcode.CLOSE:
                self._log.set_close(
                    "server", WebSocketClose.from_payload(frame.payload)
                )
            await self._write(wire)
        return recorded

    async def _write(self, data: bytes) -> None:
        writer = self._writer
        step = self._options.write_chunk_size
        try:
            if step is None or len(data) <= step:
                writer.write(data)
                await writer.drain()
                return
            view = memoryview(data)
            for start in range(0, len(view), step):
                writer.write(view[start : start + step])
                await writer.drain()
        except OSError as exc:
            self._end("client", reset=isinstance(exc, ConnectionResetError))
            raise WebSocketEnded("the client connection was lost") from exc

    def _eof_reason(self) -> CloseReason:
        return "websocket_close" if self._closing.is_set() else "client"

    async def _read_loop(self) -> None:
        decoder = FrameDecoder()
        try:
            while await self._read_once(decoder):
                pass
        except WebSocketEnded:
            return
        except Exception:
            LOG.exception("Error in WebSocket reader loop")
            self._end("error")

    async def _read_once(self, decoder: FrameDecoder) -> bool:
        """Read and handle one chunk; return whether to keep reading."""
        try:
            data = await stream.read(self._reader, self._options.max_read)
        except OSError as exc:
            self._end(
                self._eof_reason(),
                reset=isinstance(exc, ConnectionResetError),
            )
            return False
        if not data:
            self._end(self._eof_reason())
            return False
        if self._consumed is not None:
            self._consumed.extend(data)
        for decoded in decoder.feed(data):
            await self._on_frame(decoded)
            if self._ended:
                return False
        header = decoder.partial_header
        if self._options.strict and header is not None:
            violation = self._assembler.check_header(
                header
            ) or self._assembler.check_partial(header, decoder)
            if violation is not None:
                await self._violated(violation)
                return False
        return True

    async def _violated(self, violation: WebSocketViolation) -> None:
        self._log.add_violation(violation)
        await self._fail(violation.close_code, "protocol_error")

    async def _on_frame(self, decoded: DecodedFrame) -> None:
        recorded = self._record_frame(
            decoded.frame, "client", decoded.wire_raw_bytes
        )
        self._frames.put(recorded)
        result = self._assembler.receive(recorded, decoded.header)
        for violation in result.violations:
            self._log.add_violation(violation)
        if result.close is not None:
            self._log.set_close("client", result.close)
        if result.violations and self._options.strict:
            await self._fail(result.violations[0].close_code, "protocol_error")
            return
        for message in result.messages:
            self._log.add_message(message)
            self._messages.put(message.message)
        frame = decoded.frame
        if result.close is not None:
            await self._on_close(result.close)
        elif frame.opcode == Opcode.PING and self.auto_pong:
            await self._send_frame(WebSocketFrame(Opcode.PONG, frame.payload))

    async def _on_close(self, close: WebSocketClose) -> None:
        self._client_closed.set()
        if self._closing.is_set() or not self.auto_close:
            # A close() in progress ends the session once it sees this.
            return
        await self._let_handler_catch_up()
        echo = b"" if close.code is None else close.code.to_bytes(2, "big")
        await self._send_frame(WebSocketFrame(Opcode.CLOSE, echo))
        self._end("websocket_close")


def _data_size(message: WebSocketMessage) -> int:
    return len(message.data)


def _fragments(
    opcode: Opcode,
    payload: bytes,
    fragment_size: int | None,
    compressed: bool,
) -> list[WebSocketFrame]:
    """Split a message payload into data frames."""
    if fragment_size is None or len(payload) <= fragment_size:
        return [WebSocketFrame(opcode, payload, rsv1=compressed)]
    pieces = [
        payload[start : start + fragment_size]
        for start in range(0, len(payload), fragment_size)
    ]
    last = len(pieces) - 1
    return [
        WebSocketFrame(
            opcode if index == 0 else Opcode.CONTINUATION,
            piece,
            fin=index == last,
            rsv1=compressed and index == 0,
        )
        for index, piece in enumerate(pieces)
    ]
