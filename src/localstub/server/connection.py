"""One HTTP connection: its loop, its state, and its stream wrappers.

``HTTPConnection`` serves requests on one connection until it closes,
then publishes exactly one ``ConnectionClosed`` event.  It owns the
close decision through ``ConnectionState``: the first decision wins,
whichever side closed and for whatever reason, so later faults,
disconnects, or shutdown cannot overwrite it.

The server constructs a connection for every accepted or handed-off
stream and stays the source of truth for everything a test can change
at runtime.  The loop reads that configuration per request through
the callables in ``RequestPipeline``.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import sys
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any

from localstub.forward import response_allows_keep_alive
from localstub.http import stream
from localstub.http.connection import (
    ConnectionRequest,
    should_close_connection,
)
from localstub.http.exchange import ClosePhase, CloseReason, ConnectionClosed
from localstub.http.headers import HeaderItem
from localstub.http.request import (
    AsyncRequestParser,
    HTTPRequestHeaders,
    ParsedRequest,
    ParseOutcome,
    ParseStop,
    RecordedHTTPRequest,
)
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HTTPResponse
from localstub.http.utils import (
    headers_to_headers,
    serialize_header_line,
    status_phrase,
)
from localstub.middleware import (
    CloseConnection,
    CloseDuringRequest,
    ConnectionMeta,
    ForwardProxyResponse,
    HeaderContext,
    HeaderDecision,
    ResponderContext,
    ResponseSpec,
    SenderContext,
    SendResult,
    ServerServices,
    ensure_response_spec,
)
from localstub.recording import (
    DEFAULT_COALESCE_SIZE,
    BoundedByteBuffer,
    TrafficRecorder,
)
from localstub.server.transmission import (
    AbortTransmission,
    ImmediateTransmission,
    TransmissionStrategy,
)

LOG = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]
HeaderApp = Callable[[HeaderContext], Awaitable[HeaderDecision]]
ResponderApp = Callable[[ResponderContext], Awaitable[ResponseSpec]]
SenderApp = Callable[[SenderContext, ResponseSpec], Awaitable[SendResult]]
CaptureContext = Callable[[ResponderContext], None]


class ConnectionLost(Exception):
    """Raised at a write site after a transport failure was classified.

    The close has already been decided as ``client`` by the time this
    is raised, so the loop only needs to stop; it is not an error in
    user code and is logged at debug level.
    """


def pack_linger_option(*, platform: str = sys.platform) -> bytes:
    """Pack the ``SO_LINGER`` value that turns close into a TCP reset.

    The option is a C struct with no Python constant: Linux and macOS
    define it as two ints, Windows as two unsigned shorts, and the
    kernel rejects the wrong size.
    """
    layout = "HH" if platform == "win32" else "ii"
    return struct.pack(layout, 1, 0)


@lru_cache(maxsize=128)
def _serialize_response_head(
    status: int,
    headers: tuple[HeaderItem, ...],
) -> bytes:
    reason = status_phrase(status, "UNKNOWN")
    head = bytearray(f"HTTP/1.1 {status} {reason}\r\n".encode("ascii"))
    for name, value in headers:
        head.extend(serialize_header_line(name, value))
    head.extend(b"\r\n")
    return bytes(head)


def _normalize_body(body_obj: bytes | str) -> bytes:
    if isinstance(body_obj, bytes):
        return body_obj
    return body_obj.encode("utf-8")


def _is_interim(status: int) -> bool:
    """Whether ``status`` is an interim response a final one follows.

    101 is informational but final: the connection switches protocols
    right after it and no HTTP/1.1 response follows (RFC 9110 §15.2.2).
    """
    return 100 <= status < 200 and status != 101


def _body_allowed(method: str | None, status: int) -> bool:
    """Whether a response to ``method`` with ``status`` carries a body.

    A response to HEAD and any 1xx, 204, or 304 response has no body
    on the wire whatever its headers say (RFC 9112 §6.3); writing one
    would be read as the start of the next response.
    """
    if method is not None and method.upper() == "HEAD":
        return False
    return not (100 <= status < 200 or status in {204, 304})


def _with_connection_token(
    items: list[HeaderItem],
    token: str,
) -> list[HeaderItem]:
    """Ensure ``token`` is listed in the ``Connection`` header.

    An existing header gets the token appended rather than a second
    header being emitted; an absent header is added.
    """
    for index, (name, value) in enumerate(items):
        if name.lower() != "connection":
            continue
        tokens = {part.strip().lower() for part in value.split(",")}
        if token not in tokens:
            items[index] = (name, f"{value}, {token}")
        return items
    items.append(("Connection", token))
    return items


@dataclass(frozen=True)
class KeepAlivePolicy:
    """How the server treats a connection between requests.

    ``timeout`` is how long the server waits after a response for the
    first byte of the next request; ``None`` waits forever and ``0.0``
    closes right after every response.  ``max_requests`` closes after
    that many completed final responses.  ``reset`` makes timeout
    closures abortive.  ``advertise`` adds ``Keep-Alive`` hints to
    locally constructed responses that allow reuse.
    """

    timeout: float | None = None
    max_requests: int | None = None
    reset: bool = False
    advertise: bool = False

    def __post_init__(self) -> None:
        if self.timeout is not None and self.timeout < 0:
            raise ValueError("timeout must be non-negative or None")
        if self.max_requests is not None and self.max_requests < 1:
            raise ValueError("max_requests must be at least 1 or None")

    @property
    def closes_after_response(self) -> bool:
        """Whether a zero timeout closes right after every response."""
        return self.timeout is not None and self.timeout <= 0

    def keep_alive_hint(self) -> str | None:
        """The ``Keep-Alive`` header value to advertise, if any.

        The timeout is advertised only when it is an integral number
        of seconds; unset parameters are omitted, and ``None`` means
        no header at all.
        """
        if not self.advertise:
            return None
        parts: list[str] = []
        if self.timeout is not None and float(self.timeout).is_integer():
            parts.append(f"timeout={int(self.timeout)}")
        if self.max_requests is not None:
            parts.append(f"max={self.max_requests}")
        if not parts:
            return None
        return ", ".join(parts)


@dataclass(frozen=True)
class RequestPipeline:
    """Per-request configuration lookups the connection loop uses.

    Each callable is invoked when its stage runs, so changes made to
    the server between requests take effect on the next request.  The
    ``header`` builder returns ``None`` when no header middleware is
    registered, which lets the loop parse the request in one step.
    The ``sender`` builder receives the loop's terminal and returns
    ``None`` when no sender middleware wraps it.
    """

    header: Callable[[], HeaderApp | None]
    responder: Callable[[CaptureContext | None], ResponderApp]
    sender: Callable[[SenderApp], SenderApp | None]
    transmission: Callable[[], TransmissionStrategy]
    keep_alive: Callable[[], KeepAlivePolicy]


@dataclass
class ConnectionState:
    """Everything a connection remembers about itself.

    ``phase`` follows the request lifecycle so a decision made from
    outside the loop, such as shutdown, can report where the connection
    was.  ``interim_responses`` (1xx other than 101) and
    ``final_response`` are reset at the start of each request; the
    counters and ``closed`` live for the connection.  ``received`` and
    ``sent`` are the bounded buffers behind the server's connection
    byte getters, or ``None`` when that retention is disabled.
    """

    client: tuple[str, int] | None
    phase: ClosePhase = "idle"
    requests_completed: int = 0
    bytes_read: int = 0
    bytes_consumed: int = 0
    bytes_written: int = 0
    interim_responses: list[RecordedHTTPResponse] = field(
        default_factory=list[RecordedHTTPResponse]
    )
    final_response: RecordedHTTPResponse | None = None
    closed: ConnectionClosed | None = None
    received: BoundedByteBuffer | None = None
    sent: BoundedByteBuffer | None = None

    def decide_close(
        self,
        reason: CloseReason,
        phase: ClosePhase,
        *,
        reset: bool,
        timestamp: datetime,
    ) -> ConnectionClosed:
        """Record the close decision if none exists; return it.

        The counters are snapshotted at the first call; a later call
        is a no-op that returns the existing event.
        """
        if self.closed is None:
            self.closed = ConnectionClosed(
                client=self.client,
                reason=reason,
                phase=phase,
                reset=reset,
                requests_completed=self.requests_completed,
                bytes_read=self.bytes_read,
                bytes_consumed=self.bytes_consumed,
                bytes_written=self.bytes_written,
                timestamp=timestamp,
            )
        return self.closed


class ConsumedByteSink:
    """Wire sink that counts consumed bytes and retains them.

    The parser feeds every byte it attributes to a request through
    here, so the counter and the retained buffer see the same bytes at
    the same call site.
    """

    def __init__(self, state: ConnectionState) -> None:
        self._state = state

    def extend(self, data: bytes | bytearray) -> None:
        self._state.bytes_consumed += len(data)
        if self._state.received is not None:
            self._state.received.extend(data)


class CountingStreamReader:
    """Read-side counterpart of ``RecordingStreamWriter``.

    Wraps the accepted reader and feeds ``bytes_read`` on every read
    that reaches the wrapped reader.  Those reads go through
    ``localstub.http.stream``, so read-ahead an earlier consumer left
    on the wrapped reader, such as a pipelined request that
    ``HTTPRequestReader`` read past before the handoff, is served in
    stream order and counted once.  Bytes replayed against the
    wrapper itself never reach it, so they are counted once as well.
    No await separates the inner read returning from the increment,
    so a cancelled read neither loses a byte nor counts one twice.
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        state: ConnectionState,
    ) -> None:
        self._reader = reader
        self._state = state

    async def read(self, n: int = -1) -> bytes:
        data = await stream.read(self._reader, n)
        self._state.bytes_read += len(data)
        return data


class RecordingStreamWriter:
    """Wrapper that records the current response and connection bytes.

    The current response is retained the same way ``BoundedByteBuffer``
    retains bytes: a write of at least ``coalesce_size`` bytes is kept
    as a chunk of its own without a copy, and smaller writes accumulate
    in a pending ``bytearray`` that is sealed into a chunk once it
    reaches ``coalesce_size``.  A body sent in one write is therefore
    returned without copying, and a body sent one byte at a time is
    joined from a bounded number of chunks rather than one per write.
    The logic is inlined here rather than delegated because this is the
    hot path for every byte the server sends.

    When ``state`` is given, every write also feeds its
    ``bytes_written`` counter at the same call site as ``sent_buffer``.
    """

    def __init__(
        self,
        writer: asyncio.StreamWriter,
        sent_buffer: BoundedByteBuffer | None = None,
        *,
        coalesce_size: int = DEFAULT_COALESCE_SIZE,
        state: ConnectionState | None = None,
    ) -> None:
        if coalesce_size < 1:
            raise ValueError(
                f"coalesce_size must be at least 1, got {coalesce_size}"
            )
        self._writer = writer
        self._sent_buffer = sent_buffer
        self._coalesce_size = coalesce_size
        self._state = state
        self._chunks: list[bytes] = []
        self._pending = bytearray()

    @property
    def bytes_sent(self) -> bytes:
        """Return bytes written for the current response."""
        if not self._pending:
            return b"".join(self._chunks)
        return b"".join((*self._chunks, self._pending))

    def start_response(self) -> None:
        """Start recording a new response."""
        self._chunks.clear()
        self._pending.clear()

    def write(self, data: bytes) -> None:
        if len(data) >= self._coalesce_size:
            if self._pending:
                self._chunks.append(bytes(self._pending))
                self._pending.clear()
            self._chunks.append(data)
        elif data:
            self._pending += data
            if len(self._pending) >= self._coalesce_size:
                self._chunks.append(bytes(self._pending))
                self._pending.clear()
        if self._sent_buffer is not None:
            self._sent_buffer.extend(data)
        if self._state is not None:
            self._state.bytes_written += len(data)
        self._writer.write(data)

    def writelines(self, data: Iterable[bytes]) -> None:
        for chunk in data:
            self.write(chunk)

    async def drain(self) -> None:
        return await self._writer.drain()

    def write_eof(self) -> None:
        self._writer.write_eof()

    def is_closing(self) -> bool:
        return self._writer.is_closing()

    def close(self) -> None:
        self._writer.close()

    def abort(self) -> None:
        """Close the transport immediately, discarding buffered data."""
        self._writer.transport.abort()

    def reset(self) -> None:
        """Request an abortive TCP close, even after writes have drained."""
        sock = self.get_extra_info("socket")
        if sock is not None:
            try:
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, pack_linger_option()
                )
            except OSError:
                LOG.warning(
                    "Could not request a TCP reset; the client may observe "
                    "EOF instead",
                    exc_info=True,
                )
        self.abort()

    async def wait_closed(self) -> None:
        """Wait for the transport to close.

        ``StreamWriter.wait_closed()`` awaits a future the protocol
        shares with every other waiter on the same stream, and
        cancelling a task that awaits it cancels that future for all
        of them, including a proxy that owns the stream.  The wait is
        shielded so a cancelled connection loop is the only party that
        sees the cancellation.
        """
        await asyncio.shield(self._writer.wait_closed())

    def get_extra_info(self, name: str, default: Any | None = None) -> Any:
        return self._writer.get_extra_info(name, default)


@dataclass(frozen=True)
class _ConnectionWriter:
    """Stream operations attributed to the client by the connection."""

    write_bytes: Callable[[bytes], None]
    drain_stream: Callable[[], Awaitable[None]]
    close_stream: Callable[[], None]
    wait_for_close: Callable[[], Awaitable[None]]
    reset_stream: Callable[[], None]

    def write(self, data: bytes) -> None:
        self.write_bytes(data)

    async def drain(self) -> None:
        await self.drain_stream()

    def close(self) -> None:
        self.close_stream()

    def reset(self) -> None:
        self.reset_stream()

    async def wait_closed(self) -> None:
        await self.wait_for_close()


@dataclass(frozen=True)
class _Reading:
    """A request read to its message boundary, ready to respond to."""

    request: RecordedHTTPRequest
    state: dict[str, Any]
    request_timestamp: datetime
    received_monotonic: float
    responded: bool


class HTTPConnection:
    """Serve one HTTP connection and publish its close event."""

    def __init__(
        self,
        *,
        reader: CountingStreamReader,
        writer: RecordingStreamWriter,
        state: ConnectionState,
        pipeline: RequestPipeline,
        recorder: TrafficRecorder,
        services: ServerServices,
        sleep: Sleep,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._state = state
        self._pipeline = pipeline
        self._recorder = recorder
        self._services = services
        self._sleep = sleep
        self._sink = ConsumedByteSink(state)
        self._policy = KeepAlivePolicy()
        self._child: asyncio.Task[None] | None = None
        self._finalized = False
        self._sending: HTTPResponse | None = None
        self._final_send_close = False

    @property
    def state(self) -> ConnectionState:
        return self._state

    async def run(self) -> None:
        """Serve requests until the connection closes, then publish
        exactly one ConnectionClosed.

        The loop runs in a child task so ``shutdown()`` can interrupt
        whatever it is waiting on without cancelling this coroutine,
        which returns normally to its caller.  A cancellation that
        reaches this coroutine itself decides ``shutdown``, cancels the
        loop, finalizes, and re-raises.  The event is published as soon
        as the loop has finished; only then does this coroutine wait for
        the transport to close, which over TLS can depend on the peer.
        """
        if self._state.closed is not None:
            await self._finalize()
            return
        child = asyncio.create_task(self._serve())
        self._child = child
        try:
            await asyncio.wait({child})
        except asyncio.CancelledError:
            self._decide("shutdown", self._state.phase)
            child.cancel()
            await asyncio.wait({child})
            self._writer.abort()
            raise
        finally:
            await self._finalize()

    def shutdown(self) -> None:
        """Decide shutdown at the current phase and interrupt the loop.

        An earlier decision is preserved.  The loop's task is never
        cancelled from inside itself: a handler that shuts the server
        down only records the decision, and the loop exits after the
        current request.  Once the loop has finished, only the transport
        close remains, and that is aborted rather than left waiting on
        the peer.
        """
        self._decide("shutdown", self._state.phase)
        child = self._child
        if child is None or self.owns_current_task():
            return
        if child.done():
            self._writer.abort()
            return
        child.cancel()

    def owns_current_task(self) -> bool:
        """Whether the calling code runs inside this connection's loop."""
        try:
            current = asyncio.current_task()
        except RuntimeError:
            return False
        return self._child is not None and self._child is current

    def finalize(self) -> None:
        """Decide a close if none exists, close the writer, and publish.

        Idempotent; ``run()`` calls it, and the server calls it for a
        connection whose loop never starts.
        """
        if self._finalized:
            return
        self._finalized = True
        try:
            event = self._decide("error", self._state.phase)
        finally:
            # The writer closes even when deciding the event raises, so
            # a listener waiting on its connections is never stranded.
            if self._state.closed is None and not self._writer.is_closing():
                self._writer.close()
        if not self._writer.is_closing():
            self._start_close(event)
        self._recorder.record_connection_closed(event)

    async def _finalize(self) -> None:
        """Publish the event, then wait for the transport to close.

        Publishing comes first so a listener sees the event even when
        finishing the close depends on the peer, as a TLS shutdown does
        for a pooled client that is not reading.
        """
        self.finalize()
        try:
            await self._writer.wait_closed()
        except asyncio.CancelledError:
            self._writer.abort()
            raise
        except Exception:
            LOG.debug("Failed to close client writer", exc_info=True)

    def _decide(
        self,
        reason: CloseReason,
        phase: ClosePhase,
        *,
        reset: bool = False,
    ) -> ConnectionClosed:
        state = self._state
        if state.closed is not None:
            return state.closed
        return state.decide_close(
            reason,
            phase,
            reset=reset,
            timestamp=self._services.timestamp_provider.now(),
        )

    async def _serve(self) -> None:
        state = self._state
        try:
            while state.closed is None:
                self._policy = self._pipeline.keep_alive()
                if not await self._await_first_byte():
                    break
                self._policy = self._pipeline.keep_alive()
                reading = await self._read_request()
                if reading is None:
                    break
                if await self._handle_request(reading):
                    break
            if state.closed is not None and not self._writer.is_closing():
                self._start_close(state.closed)
        except ConnectionLost:
            LOG.debug("Client connection lost", exc_info=True)
        except Exception:
            self._decide("error", state.phase)
            LOG.exception("Error in AsyncHTTPTestServer handler")

    async def _await_first_byte(self) -> bool:
        """Wait for the next request's first byte.

        Returns ``False`` once the connection is to close: the keep-alive
        timer fired, or the client closed or reset while idle.  Otherwise
        the byte is returned to the stream for the parser.
        """
        state = self._state
        state.phase = "idle"
        timeout = self._policy.timeout if state.requests_completed else None
        try:
            if timeout is None or timeout <= 0:
                first = await stream.read(self._reader, 1)
            else:
                first = await self._race_first_byte(timeout)
        except OSError as exc:
            self._decide(
                "client", "idle", reset=isinstance(exc, ConnectionResetError)
            )
            return False
        if first is None:
            self._decide("idle_timeout", "idle", reset=self._policy.reset)
            return False
        if not first:
            self._decide("client", "idle")
            return False
        stream.unread_data(self._reader, first)
        state.phase = "request_headers"
        return True

    async def _race_first_byte(self, timeout: float) -> bytes | None:
        """Race the one-byte read against the injected sleep.

        Returns the byte, or ``None`` when the timer won.  The loser is
        cancelled before returning; cancelling a pending read loses no
        data.  If both finish in the same turn, the byte wins.
        """
        read_task = asyncio.ensure_future(stream.read(self._reader, 1))
        timer = asyncio.ensure_future(self._sleep(timeout))
        try:
            done, _ = await asyncio.wait(
                {read_task, timer}, return_when=asyncio.FIRST_COMPLETED
            )
        except asyncio.CancelledError:
            read_task.cancel()
            timer.cancel()
            await asyncio.gather(read_task, timer, return_exceptions=True)
            raise
        if read_task in done:
            timer.cancel()
            await asyncio.gather(timer, return_exceptions=True)
            return read_task.result()
        read_task.cancel()
        await asyncio.gather(read_task, return_exceptions=True)
        return None

    async def _read_request(self) -> _Reading | None:
        """Read one request, running the header chain when configured.

        Returns ``None`` when the connection is to close instead of
        responding: the read stopped short, or header middleware
        decided to close.  Partial requests whose headers completed are
        recorded before returning.
        """
        state = self._state
        state.interim_responses = []
        state.final_response = None
        self._final_send_close = False
        request_state: dict[str, Any] = {}
        parser = AsyncRequestParser()

        outcome, remaining = await parser.parse_headers(
            self._reader, self._sink
        )
        parsed = outcome.parsed
        if parsed is None:
            self._decide_read_stop(outcome)
            return None
        if outcome.stop is not ParseStop.COMPLETE:
            # Rejected at the header boundary, after the headers
            # completed: recorded like a body that stopped short, and
            # never shown to header middleware.
            return self._complete_reading(outcome, request_state)
        self._track_boundary(parsed)

        fault: CloseDuringRequest | None = None
        header_app = self._pipeline.header()
        if header_app is not None:
            ctx = self._header_context(
                parsed, outcome.wire_bytes, request_state
            )
            fault = await self._interruptible(
                parser, self._header_stage(header_app, ctx)
            )
        if fault is None:
            outcome = await self._interruptible(
                parser,
                parser.continue_parse_body(
                    self._reader, remaining, self._sink
                ),
            )
            return self._complete_reading(outcome, request_state)

        outcome = await self._interruptible(
            parser,
            parser.continue_parse_body(
                self._reader,
                remaining,
                self._sink,
                max_body_bytes=fault.after_body_bytes,
            ),
        )
        self._close_during_request(outcome, fault.reset)
        return None

    async def _interruptible[T](
        self,
        parser: AsyncRequestParser,
        step: Awaitable[T],
    ) -> T:
        """Run a read step, recording the request if the step fails.

        Shutdown interrupts the loop by cancelling it, and header
        middleware may raise.  A request whose headers already
        completed is still recorded, with the body consumed so far and
        any header-phase response, before the failure continues.
        """
        try:
            return await step
        except asyncio.CancelledError:
            self._record_interrupted(parser.snapshot(), "shutdown")
            raise
        except Exception:
            self._record_interrupted(parser.snapshot(), "error")
            raise

    def _record_interrupted(
        self,
        outcome: ParseOutcome,
        reason: CloseReason,
    ) -> None:
        state = self._state
        parsed = outcome.parsed
        assert parsed is not None
        request = self._build_request(parsed, outcome.wire_bytes)
        request_timestamp, _ = self._recorder.record_request(request)
        self._track_boundary(parsed)
        self._decide(reason, state.phase)
        self._record_exchange(
            request, request_timestamp, response=state.final_response
        )

    async def _header_stage(
        self,
        header_app: HeaderApp,
        ctx: HeaderContext,
    ) -> CloseDuringRequest | None:
        """Run the header chain; ``None`` means read the body and respond."""
        decision = await header_app(ctx)
        if decision is True:
            return None
        return self._header_close(decision)

    def _header_close(self, decision: HeaderDecision) -> CloseDuringRequest:
        if decision is False:
            return CloseDuringRequest()
        if isinstance(decision, CloseDuringRequest):
            return decision
        raise TypeError(f"Unhandled header decision: {decision!r}")

    def _header_context(
        self,
        parsed: ParsedRequest,
        header_wire: bytes,
        request_state: dict[str, Any],
    ) -> HeaderContext:
        partial = HTTPRequestHeaders(
            method=parsed.method,
            path=(
                parsed.url.decode("ascii", errors="replace")
                if parsed.url
                else None
            ),
            http_version=parsed.http_version,
            headers=headers_to_headers(parsed.headers),
            wire_raw_bytes=header_wire,
        )

        async def send(response: HTTPResponse) -> None:
            await self._send_header_phase_response(partial, response)

        return HeaderContext(
            headers=partial,
            connection=ConnectionMeta(client=self._state.client),
            services=self._services,
            send=send,
            state=request_state,
        )

    async def _send_header_phase_response(
        self,
        partial: HTTPRequestHeaders,
        response: HTTPResponse,
    ) -> None:
        """Write a response from header middleware and file it.

        A 1xx status other than 101 is an interim response; any other
        status, 101 included, is the final response for this request,
        and a second one is a protocol violation with no sensible
        recording.  A 101 closes the connection once the request is
        read, since the server does not speak the switched protocol.
        """
        state = self._state
        is_interim = _is_interim(response.status)
        close = False
        if not is_interim:
            if state.final_response is not None:
                raise RuntimeError(
                    "A final response was already sent for this request"
                )
            close = self._should_close_after(
                partial, response, self._completed_with_current()
            )
        recorded = self._write_early_response(partial, response, close=close)
        if is_interim:
            state.interim_responses.append(recorded)
        else:
            state.final_response = recorded
            self._final_send_close = close
        await self._drain()

    def _write_early_response(
        self,
        partial: HTTPRequestHeaders,
        response: HTTPResponse,
        *,
        close: bool,
    ) -> RecordedHTTPResponse:
        writer = self._writer
        writer.start_response()
        body = _normalize_body(response.body)
        headers = self._build_response_headers(response, body, close)
        wire_body = (
            body if _body_allowed(partial.method, response.status) else b""
        )
        self._write(
            _serialize_response_head(response.status, headers) + wire_body
        )
        return self._build_recorded_response(response, writer.bytes_sent)

    def _complete_reading(
        self,
        outcome: ParseOutcome,
        request_state: dict[str, Any],
    ) -> _Reading | None:
        """Record a request read to its boundary, or the partial one."""
        state = self._state
        parsed = outcome.parsed
        assert parsed is not None
        request = self._build_request(parsed, outcome.wire_bytes)
        request_timestamp, received_monotonic = self._recorder.record_request(
            request
        )
        if outcome.complete_request is not None:
            self._track_boundary(parsed)
            return _Reading(
                request=request,
                state=request_state,
                request_timestamp=request_timestamp,
                received_monotonic=received_monotonic,
                responded=state.final_response is not None,
            )
        state.phase = "request_body"
        self._decide_read_stop(outcome)
        self._record_exchange(
            request, request_timestamp, response=state.final_response
        )
        return None

    def _decide_read_stop(self, outcome: ParseOutcome) -> ConnectionClosed:
        """Decide the close for a request read that stopped short.

        ``_await_first_byte`` has already seen a byte of this request,
        so the phase is ``request_headers`` until the headers complete
        and ``request_body`` after.
        """
        phase: ClosePhase = (
            "request_body" if outcome.parsed is not None else "request_headers"
        )
        if outcome.stop is ParseStop.PARSE_ERROR:
            return self._decide("protocol_error", phase)
        if outcome.stop is ParseStop.READ_ERROR:
            reset = isinstance(outcome.error, ConnectionResetError)
            return self._decide("client", phase, reset=reset)
        return self._decide("client", phase)

    def _close_during_request(
        self,
        outcome: ParseOutcome,
        reset: bool,
    ) -> None:
        """Record and start the close header middleware decided on."""
        state = self._state
        parsed = outcome.parsed
        assert parsed is not None
        request = self._build_request(parsed, outcome.wire_bytes)
        request_timestamp, _ = self._recorder.record_request(request)
        self._track_boundary(parsed)
        event = self._decide("request_read", state.phase, reset=reset)
        self._record_exchange(
            request, request_timestamp, response=state.final_response
        )
        self._start_close(event)

    def _track_boundary(self, parsed: ParsedRequest) -> None:
        """Advance the phase once the request's headers have been read.

        A request is counted the first time it is seen at its message
        boundary, whether that is at the end of its headers or of its
        body, so a close decided at any later point reports both the
        ``response`` phase and the incremented counter.
        """
        state = self._state
        if not parsed.is_complete:
            state.phase = "request_body"
        elif state.phase != "response":
            state.requests_completed += 1
            state.phase = "response"

    def _completed_with_current(self) -> int:
        """Requests completed on this connection once the current one is."""
        state = self._state
        if state.phase == "response":
            return state.requests_completed
        return state.requests_completed + 1

    def _build_request(
        self,
        parsed: ParsedRequest,
        wire_bytes: bytes,
    ) -> RecordedHTTPRequest:
        return RecordedHTTPRequest.from_parsed(
            parsed,
            wire_bytes,
            client=self._state.client,
        )

    def _record_exchange(
        self,
        request: RecordedHTTPRequest,
        request_timestamp: datetime,
        *,
        response: RecordedHTTPResponse | None,
    ) -> None:
        state = self._state
        self._recorder.record_exchange(
            request=request,
            response=response,
            request_timestamp=request_timestamp,
            interim_responses=tuple(state.interim_responses),
            closed=state.closed,
        )
        self._sending = None

    async def _handle_request(self, reading: _Reading) -> bool:
        """Respond to a complete request; return whether to close.

        The exchange is always recorded, with the close event when the
        close was decided while the exchange was in progress.
        """
        state = self._state
        state.phase = "response"
        self._sending = None
        try:
            if reading.responded:
                result = SendResult(
                    recorded=state.final_response,
                    should_close=self._final_send_close,
                )
            else:
                result = await self._respond(reading)
        except asyncio.CancelledError:
            self._record_failed_exchange(reading, "shutdown")
            raise
        except Exception:
            self._record_failed_exchange(reading, "error")
            raise
        if result.closed is not None:
            should_close = True
        else:
            state.phase = "after_response"
            should_close = self._apply_close_policy(result.should_close)
        self._record_exchange(
            reading.request,
            reading.request_timestamp,
            response=result.recorded,
        )
        return should_close

    def _record_failed_exchange(
        self,
        reading: _Reading,
        reason: CloseReason,
    ) -> None:
        """Record the exchange of a request whose handling did not finish."""
        self._decide(reason, self._state.phase)
        self._record_exchange(
            reading.request,
            reading.request_timestamp,
            response=self._partial_response(),
        )

    def _partial_response(self) -> RecordedHTTPResponse | None:
        """The response being written when the exchange failed, if any."""
        if self._sending is None or not self._writer.bytes_sent:
            return None
        return self._build_recorded_response(
            self._sending, self._writer.bytes_sent
        )

    def _apply_close_policy(self, should_close: bool) -> bool:
        """Decide any policy close after a successful response write.

        A fault, transport failure, or shutdown decided during the
        response takes precedence and is never overwritten.
        """
        state = self._state
        policy = self._policy
        if state.closed is not None:
            return True
        if (
            policy.max_requests is not None
            and state.requests_completed >= policy.max_requests
        ):
            self._decide("max_requests", "after_response")
        elif policy.closes_after_response:
            self._decide("idle_timeout", "after_response", reset=policy.reset)
        elif should_close:
            self._decide("connection_close", "after_response")
        else:
            return False
        return True

    async def _respond(self, reading: _Reading) -> SendResult:
        request = reading.request
        sender_request = request

        def capture(ctx: ResponderContext) -> None:
            nonlocal sender_request
            sender_request = ctx.request

        sender = self._pipeline.sender(self._send_response)
        responder = self._pipeline.responder(capture)
        connection = ConnectionMeta(client=self._state.client)
        response_spec = await responder(
            ResponderContext(
                request=request,
                connection=connection,
                services=self._services,
                state=reading.state,
                received_monotonic=reading.received_monotonic,
            )
        )
        sender_ctx = SenderContext(
            request=sender_request,
            connection=connection,
            services=self._services,
            state=reading.state,
        )
        if sender is None:
            return await self._send_response(sender_ctx, response_spec)
        return await sender(sender_ctx, response_spec)

    async def _send_response(
        self,
        ctx: SenderContext,
        response: ResponseSpec,
    ) -> SendResult:
        """Terminal of the sender chain: put one response spec on the wire."""
        response = ensure_response_spec(response)
        if isinstance(response, CloseConnection):
            return await self._close_response(response)

        writer = self._writer
        writer.start_response()
        if isinstance(response, ForwardProxyResponse):
            return await self._relay(ctx, response)

        self._sending = response
        should_close, abort = await self._write_response(response, ctx.request)
        recorded = self._build_recorded_response(response, writer.bytes_sent)
        if abort is None:
            return SendResult(recorded=recorded, should_close=should_close)
        event = self._decide(
            "response_aborted", "response_body", reset=abort.reset
        )
        self._start_close(event)
        return SendResult(recorded=recorded, should_close=True, closed=event)

    async def _close_response(self, spec: CloseConnection) -> SendResult:
        if spec.delay > 0:
            await self._sleep(spec.delay)
        event = self._decide("close_response", "response", reset=spec.reset)
        self._start_close(event)
        return SendResult(recorded=None, should_close=True, closed=event)

    async def _relay(
        self,
        ctx: SenderContext,
        response: ForwardProxyResponse,
    ) -> SendResult:
        writer = self._writer

        def write(data: bytes) -> None:
            self._state.phase = "response_body"
            self._write(data)

        def reset() -> None:
            event = self._decide(
                "response_aborted", "response_body", reset=True
            )
            self._start_close(event)

        client_writer = _ConnectionWriter(
            write_bytes=write,
            drain_stream=self._drain,
            close_stream=writer.close,
            wait_for_close=writer.wait_closed,
            reset_stream=reset,
        )
        result = await response.forwarder.forward_and_relay(
            host=response.host,
            port=response.port,
            request_wire_bytes=response.request_wire_bytes,
            client_writer=client_writer,
            request_method=response.request_method,
            upstream_tls=response.upstream_tls,
        )
        if result is None:
            return await self._send_response(
                ctx, HTTPResponse.text("Bad Gateway", status=502)
            )
        recorded = result.to_recorded_response(
            wire_raw_bytes=writer.bytes_sent,
        )
        should_close = writer.is_closing() or not response_allows_keep_alive(
            ctx.request, result
        )
        return SendResult(
            recorded=recorded,
            should_close=should_close,
            closed=self._state.closed,
        )

    async def _write_response(
        self,
        response: HTTPResponse,
        request: RecordedHTTPRequest,
    ) -> tuple[bool, AbortTransmission | None]:
        writer = self._writer
        body = _normalize_body(response.body)
        wire_body = (
            body if _body_allowed(request.method, response.status) else b""
        )
        should_close = self._should_close_after(
            request, response, self._state.requests_completed
        )
        headers = self._build_response_headers(response, body, should_close)
        head = _serialize_response_head(response.status, headers)

        strategy = self._pipeline.transmission()
        self._state.phase = "response_body"
        if isinstance(strategy, ImmediateTransmission):
            self._write(head + wire_body)
            await self._drain()
            return should_close, None

        self._write(head)
        client_writer = _ConnectionWriter(
            write_bytes=self._write,
            drain_stream=self._drain,
            close_stream=writer.close,
            wait_for_close=writer.wait_closed,
            reset_stream=writer.reset,
        )
        abort = await strategy.write_body(client_writer, wire_body)
        return should_close, abort

    def _write(self, data: bytes) -> None:
        try:
            self._writer.write(data)
        except OSError as exc:
            raise self._connection_lost(exc) from exc

    async def _drain(self) -> None:
        try:
            await self._writer.drain()
        except OSError as exc:
            raise self._connection_lost(exc) from exc

    def _connection_lost(self, exc: OSError) -> ConnectionLost:
        """Classify a transport failure at a write site."""
        self._decide(
            "client",
            self._state.phase,
            reset=isinstance(exc, ConnectionResetError),
        )
        return ConnectionLost(str(exc))

    def _should_close_after(
        self,
        request: ConnectionRequest,
        response: HTTPResponse,
        completed: int,
    ) -> bool:
        """Whether the connection closes once ``response`` is written.

        ``completed`` is the number of requests completed on this
        connection once the current one is counted.
        """
        if should_close_connection(
            request,
            response_headers=response.headers,
            response_status=response.status,
        ):
            return True
        max_requests = self._policy.max_requests
        return max_requests is not None and completed >= max_requests

    def _build_response_headers(
        self,
        response: HTTPResponse,
        body: bytes,
        should_close: bool,
    ) -> tuple[HeaderItem, ...]:
        """Build the complete response header items, in order."""
        items = list(response.headers.items())
        header_names = {name.lower() for name, _ in items}
        is_informational = 100 <= response.status < 200

        if is_informational or response.status == 204:
            items = [
                (name, value)
                for name, value in items
                if name.lower() != "content-length"
            ]
        elif "content-length" not in header_names:
            items.append(("Content-Length", str(len(body))))

        if should_close:
            items = _with_connection_token(items, "close")
        elif not is_informational:
            hint = self._policy.keep_alive_hint()
            if hint is not None:
                items.append(("Keep-Alive", hint))
                items = _with_connection_token(items, "keep-alive")

        return tuple(items)

    def _build_recorded_response(
        self,
        response: HTTPResponse,
        wire_bytes: bytes,
    ) -> RecordedHTTPResponse:
        return RecordedHTTPResponse(
            response=HTTPResponse(
                status=response.status,
                headers=response.headers,
                body=_normalize_body(response.body),
            ),
            reason=status_phrase(response.status),
            wire_raw_bytes=wire_bytes,
        )

    def _start_close(self, event: ConnectionClosed) -> None:
        """Begin closing the writer the way ``event`` calls for.

        A reset arms the linger option and aborts.  Shutdown does not
        wait for the peer: ``close()`` queues any TLS close_notify and
        ``abort()`` then drops the socket instead of holding the TLS
        shutdown handshake open for a peer that is not reading; on a
        flushed TCP transport the abort is a no-op.  The close is not
        awaited here; ``run()`` waits for it once the event is published.
        """
        writer = self._writer
        if event.reset:
            writer.reset()
            return
        writer.close()
        if event.reason == "shutdown":
            writer.abort()
