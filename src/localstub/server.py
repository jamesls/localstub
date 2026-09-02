from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import (
    Any,
    Protocol,
    Self,
)

from localstub.forward import RawForwarder, response_allows_keep_alive
from localstub.http.client import HTTPClient
from localstub.http.connection import should_close_connection
from localstub.http.exchange import RecordedExchange
from localstub.http.headers import HeaderItem
from localstub.http.request import (
    AsyncRequestParser,
    HTTPRequestHeaders,
    ParsedRequest,
    RecordedHTTPRequest,
)
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HeadersLike, HTTPResponse
from localstub.http.utils import (
    headers_to_headers,
    maybe_await,
    serialize_header_line,
    status_phrase,
)
from localstub.middleware import (
    ConnectionMeta,
    ForwardProxyResponse,
    HeaderContext,
    HeaderMiddleware,
    HeaderNext,
    ResponderContext,
    ResponderMiddleware,
    ResponseSpec,
    SenderContext,
    SenderMiddleware,
    SendResult,
    ServerServices,
    SystemTimestampProvider,
    TimestampProvider,
    compose_headers,
    compose_responder,
    compose_sender,
)
from localstub.middleware.builtins import (
    BuiltinMiddlewares,
    ForwardProxyMiddleware,
    HandlerMiddleware,
    RawForwardProxyMiddleware,
    ResponseSequenceMiddleware,
    RouterMiddleware,
    ThrottleMiddleware,
    ThrottleResponseFunc,
    default_throttle_response,
)
from localstub.recording import (
    DEFAULT_MAX_CONNECTION_BYTES,
    DEFAULT_RECORDING_BUFFER_SIZE,
    BoundedByteBuffer,
    TrafficRecorder,
)
from localstub.router import ResponderHandler, Router
from localstub.throttle import (
    Clock,
    MonotonicClock,
    ThrottleDecision,
    ThrottleKeyFunc,
    TokenBucketThrottler,
)

LOG = logging.getLogger(__name__)
ThrottleResponse = HTTPResponse | ThrottleResponseFunc

# Event-loop turns aclose() yields after pausing accepts, so already-accepted
# connections reach _client_connected and get torn down. Turning an accepted
# socket into that callback took up to 5 turns when measured on CPython 3.12
# (worst case: a connection made by a blocking connect()); this leaves
# headroom above that. These are bare sleep(0) yields, so unused turns cost
# nothing beyond a few trips through the loop.
_SHUTDOWN_DRAIN_TURNS = 8


def _pause_server_accepts(server: asyncio.Server) -> None:
    loop = asyncio.get_running_loop()
    try:
        for listener in server.sockets:
            loop.remove_reader(listener.fileno())
    except NotImplementedError:
        # Proactor loops attach accepted transports in the accept callback, so
        # they have no selector task that can be stranded by Server.close().
        server.close()


async def _drain_pending_accepts() -> None:
    for _ in range(_SHUTDOWN_DRAIN_TURNS):
        await asyncio.sleep(0)


def _default_throttle_key(_: RecordedHTTPRequest) -> str:
    return "global"


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


# Callback to send a response to client during header processing
SendResponse = Callable[[HTTPResponse], Awaitable[None]]

# Lifecycle hook called after headers are received, before body is read.
# Return True to continue reading body, False to stop.
OnHeadersReceived = Callable[
    [HTTPRequestHeaders, SendResponse],
    Awaitable[bool] | bool,
]


class Writer(Protocol):
    """Minimal StreamWriter interface used by transmission strategies."""

    def write(self, data: bytes) -> None: ...

    def writelines(self, data: Iterable[bytes]) -> None: ...

    async def drain(self) -> None: ...

    def write_eof(self) -> None: ...

    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...

    def is_closing(self) -> bool: ...

    def get_extra_info(self, name: str, default: Any | None = None) -> Any: ...


class RecordingStreamWriter:
    """Wrapper that records the current response and connection bytes."""

    def __init__(
        self,
        writer: asyncio.StreamWriter,
        sent_buffer: BoundedByteBuffer | None = None,
    ) -> None:
        self._writer = writer
        self._sent_buffer = sent_buffer
        self._recorded = bytearray()

    @property
    def bytes_sent(self) -> bytes:
        """Return bytes written for the current response."""
        return bytes(self._recorded)

    def start_response(self) -> None:
        """Start recording a new response."""
        self._recorded.clear()

    def write(self, data: bytes) -> None:
        self._recorded.extend(data)
        if self._sent_buffer is not None:
            self._sent_buffer.extend(data)
        self._writer.write(data)

    def writelines(self, data: Iterable[bytes]) -> None:
        for chunk in data:
            self.write(chunk)

    async def drain(self) -> None:
        return await self._writer.drain()

    def write_eof(self) -> None:
        write_eof_fn = getattr(self._writer, "write_eof", None)
        if callable(write_eof_fn):
            write_eof_fn()

    def is_closing(self) -> bool:
        return self._writer.is_closing()

    def close(self) -> None:
        self._writer.close()

    async def wait_closed(self) -> None:
        wait_closed_fn = getattr(self._writer, "wait_closed", None)
        if callable(wait_closed_fn):
            result = wait_closed_fn()
            if inspect.isawaitable(result):
                await result

    def get_extra_info(self, name: str, default: Any | None = None) -> Any:
        return self._writer.get_extra_info(name, default)


class TransmissionStrategy:
    """Protocol for controlling how response body bytes are transmitted.

    This allows tests to simulate network conditions like slow transfers,
    throttled bandwidth, etc. without changing the actual response content.
    """

    async def write_body(
        self,
        writer: Writer,
        body: bytes,
    ) -> None:
        """Write the response body to the client.

        Args:
            writer: The asyncio stream writer to write to
            body: The complete response body bytes to transmit
        """
        raise NotImplementedError


class ImmediateTransmission(TransmissionStrategy):
    """Default transmission strategy - send entire body immediately."""

    async def write_body(
        self,
        writer: Writer,
        body: bytes,
    ) -> None:
        writer.write(body)
        await writer.drain()


class ThrottledTransmission(TransmissionStrategy):
    """Throttled transmission strategy - send body in chunks with delays.

    Useful for testing client behavior with slow network connections or
    bandwidth-limited scenarios (e.g., S3 GetObject with slow transfer).
    """

    def __init__(
        self,
        chunk_size: int,
        delay: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize throttled transmission.

        Args:
            chunk_size: Number of bytes to send in each chunk
            delay: Seconds to wait between chunks
            sleep: Coroutine function used to wait between chunks,
                defaults to ``asyncio.sleep``
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size
        self.delay = delay
        self._sleep = sleep

    async def write_body(
        self,
        writer: Writer,
        body: bytes,
    ) -> None:
        offset = 0
        while offset < len(body):
            chunk = body[offset : offset + self.chunk_size]
            writer.write(chunk)
            await writer.drain()

            offset += self.chunk_size
            if offset < len(body):  # Don't delay after last chunk
                await self._sleep(self.delay)


@dataclass
class ApplyResult:
    """Result of applying a fault step."""

    body: bytes
    delay_before: float = 0.0
    drop_after: int | None = None


class FaultStep(Protocol):
    """Protocol for fault steps that mutate transmission behavior."""

    def apply(self, body: bytes) -> ApplyResult: ...


class Delay(FaultStep):
    """Delay sending the body."""

    def __init__(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        self.seconds = seconds

    def apply(self, body: bytes) -> ApplyResult:
        return ApplyResult(body=body, delay_before=self.seconds)


class DropConnection(FaultStep):
    """Close the connection after sending part of the body."""

    def __init__(self, after_bytes: int) -> None:
        if after_bytes < 0:
            raise ValueError("after_bytes must be non-negative")
        self.after_bytes = after_bytes

    def apply(self, body: bytes) -> ApplyResult:
        return ApplyResult(body=body, drop_after=self.after_bytes)


class TruncateBody(FaultStep):
    """Send only the first N bytes of the body."""

    def __init__(self, keep_bytes: int) -> None:
        if keep_bytes < 0:
            raise ValueError("keep_bytes must be non-negative")
        self.keep_bytes = keep_bytes

    def apply(self, body: bytes) -> ApplyResult:
        return ApplyResult(body=body[: self.keep_bytes])


class ByteFlip(FaultStep):
    """Flip a single byte in the body using XOR."""

    def __init__(self, offset: int, mask: int = 0xFF) -> None:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if mask < 0 or mask > 0xFF:
            raise ValueError("mask must be between 0 and 255")
        self.offset = offset
        self.mask = mask

    def apply(self, body: bytes) -> ApplyResult:
        if self.offset >= len(body):
            return ApplyResult(body=body)
        mutated = bytearray(body)
        mutated[self.offset] ^= self.mask
        return ApplyResult(body=bytes(mutated))


class FaultyTransmission(TransmissionStrategy):
    """Always-on fault injection applied during body transmission."""

    def __init__(
        self,
        faults: list[FaultStep],
        base: TransmissionStrategy | None = None,
    ) -> None:
        self._faults = faults
        self._base = base or ImmediateTransmission()

    async def write_body(
        self,
        writer: Writer,
        body: bytes,
    ) -> None:
        body_to_send = body
        total_delay = 0.0
        drop_after: int | None = None

        for fault in self._faults:
            result = fault.apply(body_to_send)
            body_to_send = result.body
            total_delay += result.delay_before
            if drop_after is None and result.drop_after is not None:
                drop_after = result.drop_after

        if total_delay > 0:
            await asyncio.sleep(total_delay)

        if drop_after is None:
            await self._base.write_body(writer, body_to_send)
            return

        to_send = body_to_send[:drop_after]
        if to_send:
            writer.write(to_send)
            await writer.drain()

        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            LOG.debug("Failed to close response writer", exc_info=True)


class AsyncHTTPTestServer:
    """Small asyncio HTTP server used for testing SDK clients.

    Features:
      * exposes .url (e.g. "http://127.0.0.1:12345/")
      * records last_request (RecordedHTTPRequest) and a list of all
        requests
      * `wire_raw_bytes` contains the *exact* bytes received, including
        chunked framing and trailers.
      * configurable static response, or plug in responder middleware.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        handler: ResponderHandler | None = None,
        default_response: HTTPResponse | None = None,
        on_headers_received: OnHeadersReceived | None = None,
        upstream_client: HTTPClient | None = None,
        raw_forwarder: RawForwarder | None = None,
        clock: Clock | None = None,
        timestamp_provider: TimestampProvider | None = None,
        recording_buffer_size: int | None = DEFAULT_RECORDING_BUFFER_SIZE,
        recorder: TrafficRecorder | None = None,
        *,
        max_connection_bytes: int | None = DEFAULT_MAX_CONNECTION_BYTES,
    ) -> None:
        if max_connection_bytes is not None and max_connection_bytes < 0:
            raise ValueError(
                "max_connection_bytes must be non-negative or None, "
                f"got {max_connection_bytes}"
            )
        self._host = host
        self._port = port
        self._server: asyncio.base_events.Server | None = None
        self._closing = False
        self._client_writers: set[asyncio.StreamWriter] = set()
        self._client_tasks: set[asyncio.Task[None]] = set()

        self._handler: ResponderHandler | None = handler
        self._default_response: HTTPResponse = (
            default_response or HTTPResponse.json({})
        )
        self.router = Router()
        self._on_headers_received = on_headers_received
        self._upstream_client = upstream_client
        self._raw_forwarder = raw_forwarder

        self._builtins = BuiltinMiddlewares()
        if raw_forwarder is not None:
            self._builtins.set(
                "raw_proxy", RawForwardProxyMiddleware(raw_forwarder)
            )
        if upstream_client is not None:
            self._builtins.set(
                "proxy", ForwardProxyMiddleware(upstream_client)
            )

        self.responder_middlewares: list[ResponderMiddleware] = []
        self.sender_middlewares: list[SenderMiddleware] = []
        self.header_middlewares: list[HeaderMiddleware] = []

        self._transmission_strategy: TransmissionStrategy = (
            ImmediateTransmission()
        )

        self._clock: Clock = clock or MonotonicClock()
        self._timestamp_provider = (
            timestamp_provider or SystemTimestampProvider()
        )
        self._services = ServerServices(
            clock=self._clock,
            timestamp_provider=self._timestamp_provider,
        )

        # Recorded history and queues are bounded so memory stays flat
        # when the server runs long enough that traffic outpaces whatever
        # is consuming the records (e.g. the CLI's forward-proxy mode).
        self._recorder = recorder or TrafficRecorder(
            recording_buffer_size,
            clock=self._clock,
            timestamp_provider=self._timestamp_provider,
        )
        self._max_connection_bytes = max_connection_bytes

        # Connection-level raw bytes tracking (keyed by client address)
        self._connection_raw_bytes_received: dict[
            tuple[str, int], BoundedByteBuffer
        ] = {}
        self._connection_raw_bytes_sent: dict[
            tuple[str, int], BoundedByteBuffer
        ] = {}

        self.host: str | None = None
        self.port: int | None = None

    @property
    def url(self) -> str:
        if self.host is None or self.port is None:
            raise RuntimeError("Server not started yet")
        return f"http://{self.host}:{self.port}/"

    @property
    def handler(self) -> ResponderHandler | None:
        return self._handler

    @handler.setter
    def handler(self, value: ResponderHandler | None) -> None:
        self._handler = value

    @property
    def default_response(self) -> HTTPResponse:
        return self._default_response

    @default_response.setter
    def default_response(self, response: HTTPResponse) -> None:
        self.set_default_response(response)

    def add_route(
        self,
        method: str,
        path: str,
        handler: ResponderHandler,
    ) -> None:
        self.router.add(method, path, handler)

    def use(self, middleware: ResponderMiddleware) -> None:
        self.responder_middlewares.append(middleware)

    def use_sender(self, middleware: SenderMiddleware) -> None:
        self.sender_middlewares.append(middleware)

    def use_headers(self, middleware: HeaderMiddleware) -> None:
        self.header_middlewares.append(middleware)

    def set_request_headers_handler(
        self,
        handler: OnHeadersReceived,
    ) -> None:
        """Set handler when client request headers are received.

        This will overwrite the `on_headers_received` value if one was
        provided when this class was instantiated.

        """
        self._on_headers_received = handler

    def set_json_response(
        self,
        obj: Any,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
    ) -> None:
        """Configure a static JSON response returned for every request."""
        self.set_default_response(
            HTTPResponse.json(obj, status=status, headers=headers)
        )

    def set_text_response(
        self,
        text: str,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
    ) -> None:
        self.set_default_response(
            HTTPResponse.text(text, status=status, headers=headers)
        )

    def set_raw_response(
        self,
        data: bytes,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
    ) -> None:
        self.set_default_response(
            HTTPResponse.raw(data, status=status, headers=headers)
        )

    def set_default_response(self, response: HTTPResponse) -> None:
        """Configure a static response returned for every request.

        Unlike set_json_response/set_text_response/set_raw_response, this
        accepts an already-constructed HTTPResponse object.

        Args:
            response: HTTPResponse object to return for all requests
        """
        self._default_response = response
        self._builtins.clear("sequence")

    def set_response_sequence(self, responses: list[HTTPResponse]) -> None:
        """Configure a sequence of responses to return in order.

        Each incoming request will consume the next response from the sequence.
        Once exhausted, falls back to handler or default_response behavior.

        This is useful for testing retry logic where you want the first N
        requests to fail and subsequent requests to succeed.

        Example:
            server.set_response_sequence([
                HTTPResponse(status=500),  # First request fails
                HTTPResponse(status=500),  # Second request fails
                HTTPResponse.json({"ok": True})  # Third request succeeds
            ])

        Args:
            responses: List of HTTPResponse objects to return in sequence
        """
        self._builtins.set("sequence", ResponseSequenceMiddleware(responses))
        # Clear default response (last one wins)
        self._default_response = HTTPResponse.json({})

    def set_transmission_strategy(
        self, strategy: TransmissionStrategy
    ) -> None:
        """Configure how response body bytes are transmitted.

        This controls the network transmission behavior (e.g., throttling,
        chunking) without changing the actual response content. Useful for
        testing client behavior under various network conditions.

        Example:
            # Simulate slow S3 GetObject response
            server.set_raw_response(large_file_bytes)
            server.set_transmission_strategy(
                ThrottledTransmission(chunk_size=8192, delay=0.1)
            )

        Args:
            strategy: TransmissionStrategy instance controlling transmission
        """
        self._transmission_strategy = strategy

    def get_connection_bytes_received(
        self, client: tuple[str, int]
    ) -> bytes | None:
        """Get retained raw bytes received from a client connection.

        Args:
            client: Tuple of (host, port) identifying the client connection

        Returns:
            Retained bytes from this client, or None if no data recorded
        """
        buf = self._connection_raw_bytes_received.get(client)
        return bytes(buf) if buf is not None else None

    def get_connection_bytes_sent(
        self, client: tuple[str, int]
    ) -> bytes | None:
        """Get retained raw bytes sent to a client connection.

        Args:
            client: Tuple of (host, port) identifying the client connection

        Returns:
            Retained bytes sent to this client, or None if no data recorded
        """
        buf = self._connection_raw_bytes_sent.get(client)
        return bytes(buf) if buf is not None else None

    def get_connection_dropped_bytes_received(
        self, client: tuple[str, int]
    ) -> int | None:
        """Get the count of received bytes dropped for a connection."""
        buf = self._connection_raw_bytes_received.get(client)
        return buf.dropped if buf is not None else None

    def get_connection_dropped_bytes_sent(
        self, client: tuple[str, int]
    ) -> int | None:
        """Get the count of sent bytes dropped for a connection."""
        buf = self._connection_raw_bytes_sent.get(client)
        return buf.dropped if buf is not None else None

    def get_request_timestamp(self, request: RecordedHTTPRequest) -> float:
        """Get the reception timestamp for a request.

        Args:
            request: The RecordedHTTPRequest object to look up.

        Returns:
            Monotonic timestamp when the request was received.

        Raises:
            ValueError: If the request is not found.  Timestamps are only
                retained for requests still in the ``requests`` history,
                which is bounded by ``recording_buffer_size``.
        """
        return self._recorder.get_request_timestamp(request)

    def clear_requests(self) -> None:
        """Clear all recorded request state.

        Resets last_request, requests list, the request queue, and
        connection-level raw bytes while preserving server configuration
        (handler, default_response, etc.).

        Useful for reusing a session-scoped test server across multiple
        tests without needing to shut down and restart the server.

        Example:
            async with AsyncHTTPTestServer() as server:
                # Test 1
                response = await client.get(server.url)
                assert len(server.requests) == 1

                # Clear state between tests
                server.clear_requests()

                # Test 2 - fresh state
                response = await client.get(server.url)
                assert len(server.requests) == 1
        """
        self._recorder.reset()
        for buf in self._connection_raw_bytes_received.values():
            buf.clear()
        for buf in self._connection_raw_bytes_sent.values():
            buf.clear()
        self._connection_raw_bytes_received.clear()
        self._connection_raw_bytes_sent.clear()
        self._builtins.reset_all()

    def set_throttle(
        self,
        *,
        rate_per_second: float,
        key: ThrottleKeyFunc | None = None,
        burst: float | None = None,
        response: ThrottleResponse | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Enable request-rate throttling for this server.

        This is request-per-second throttling enforced via a token bucket.
        When throttled, the server returns the configured throttling response.

        Args:
            rate_per_second: Token bucket refill rate (requests per second).
            key: Function mapping a request to a hashable throttle key.
            burst: Max burst capacity for each key bucket (must be >= 1).
            response: Static HTTPResponse or callable response builder used
                when a request is throttled.
            clock: Optional clock for deterministic testing.
        """
        key_fn = key or _default_throttle_key
        throttler = TokenBucketThrottler(
            rate_per_second=rate_per_second,
            key=key_fn,
            burst=burst,
            clock=clock,
        )
        response_fn = self._normalize_throttle_response(response)
        self._builtins.set(
            "throttle",
            ThrottleMiddleware(throttler=throttler, response=response_fn),
        )

    def clear_throttle(self) -> None:
        """Disable request-rate throttling."""
        self._builtins.clear("throttle")

    def _normalize_throttle_response(
        self,
        response: ThrottleResponse | None,
    ) -> ThrottleResponseFunc:
        if response is None:
            return default_throttle_response
        if isinstance(response, HTTPResponse):

            def static(
                _: RecordedHTTPRequest,
                __: ThrottleDecision,
                *,
                _response: HTTPResponse = response,
            ) -> HTTPResponse:
                return _response

            return static
        return response

    async def start(self) -> None:
        if self._server is not None:
            return

        self._closing = False
        self._server = await asyncio.start_server(
            self._client_connected,
            self._host,
            self._port,
        )
        assert self._server.sockets
        sockname = self._server.sockets[0].getsockname()
        self.host, self.port = sockname[0], sockname[1]

    async def aclose(self) -> None:
        server = self._server
        if server is None:
            return
        self._closing = True
        cancelled: asyncio.CancelledError | None = None
        try:
            # Stop selector loops from accepting more sockets without closing
            # the asyncio.Server yet. Closing it while an already-accepted
            # socket is still attaching its transport can strand the peer.
            _pause_server_accepts(server)
            drain_task = asyncio.create_task(_drain_pending_accepts())
            try:
                await asyncio.shield(drain_task)
            except asyncio.CancelledError as exc:
                # Finish the critical drain before honoring cancellation.
                await drain_task
                cancelled = exc

            server.close()
            for writer in tuple(self._client_writers):
                writer.close()
            current_task = asyncio.current_task()
            client_tasks = tuple(
                task for task in self._client_tasks if task is not current_task
            )
            for task in client_tasks:
                task.cancel()
            if client_tasks:
                await asyncio.gather(*client_tasks, return_exceptions=True)
            await server.wait_closed()
            self._client_tasks.clear()
        finally:
            server.close()
            if self._server is server:
                self._server = None
            self._closing = False

        if cancelled is not None:
            raise cancelled

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    @property
    def requests(self) -> list[RecordedHTTPRequest]:
        """All recorded requests, oldest first (bounded history)."""
        return self._recorder.requests

    @property
    def last_request(self) -> RecordedHTTPRequest | None:
        """The most recently recorded or consumed request."""
        return self._recorder.last_request

    @property
    def responses(self) -> list[RecordedHTTPResponse]:
        """All recorded responses, oldest first (bounded history)."""
        return self._recorder.responses

    @property
    def last_response(self) -> RecordedHTTPResponse | None:
        """The most recently recorded or consumed response."""
        return self._recorder.last_response

    @property
    def exchanges(self) -> list[RecordedExchange]:
        """All recorded exchanges, oldest first (bounded history)."""
        return self._recorder.exchanges

    @property
    def last_exchange(self) -> RecordedExchange | None:
        """The most recently recorded exchange."""
        return self._recorder.last_exchange

    async def next_request(
        self, timeout: float | None = None
    ) -> RecordedHTTPRequest:
        """Await and return the next request that hits this server."""
        return await self._recorder.next_request(timeout)

    async def next_response(
        self, timeout: float | None = None
    ) -> RecordedHTTPResponse:
        """Await and return the next response sent by this server."""
        return await self._recorder.next_response(timeout)

    async def next_exchange(
        self, timeout: float | None = None
    ) -> RecordedExchange:
        """Await and return the next completed request/response exchange."""
        return await self._recorder.next_exchange(timeout)

    def next_exchange_nowait(self) -> RecordedExchange | None:
        """Return the next completed exchange, or None if none is queued."""
        return self._recorder.next_exchange_nowait()

    @property
    def dropped_requests(self) -> int:
        """Requests evicted unread from the next_request() buffer."""
        return self._recorder.dropped_requests

    @property
    def dropped_responses(self) -> int:
        """Responses evicted unread from the next_response() buffer."""
        return self._recorder.dropped_responses

    @property
    def dropped_exchanges(self) -> int:
        """Exchanges evicted unread from the next_exchange() buffer."""
        return self._recorder.dropped_exchanges

    async def handle_http_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an HTTP conversation on externally-established streams.

        This method handles the full HTTP request/response cycle on streams
        that may have been established externally (e.g., by a TLS proxy
        that terminates TLS and hands off the decrypted streams).

        The server will:
        1. Parse incoming HTTP requests
        2. Record requests in .requests and .last_request
        3. Generate responses using the configured handler/routes
        4. Write responses to the client
        5. Handle connection persistence (keep-alive vs close)

        Args:
            reader: StreamReader for the client connection
            writer: StreamWriter for the client connection

        Note:
            This method is intended for external callers like TLS proxies.
            The server manages cleanup of the writer on exit.
        """
        await self._handle_client(reader, writer)

    def _init_connection_tracking(
        self, client: tuple[str, int] | None
    ) -> tuple[BoundedByteBuffer | None, BoundedByteBuffer | None]:
        """Initialize connection-level byte tracking for a client."""
        if client is None or self._max_connection_bytes == 0:
            return None, None

        if client not in self._connection_raw_bytes_received:
            self._connection_raw_bytes_received[client] = BoundedByteBuffer(
                self._max_connection_bytes
            )
        if client not in self._connection_raw_bytes_sent:
            self._connection_raw_bytes_sent[client] = BoundedByteBuffer(
                self._max_connection_bytes
            )

        return (
            self._connection_raw_bytes_received[client],
            self._connection_raw_bytes_sent[client],
        )

    def _build_responder(
        self,
        *,
        capture_ctx: Callable[[ResponderContext], None] | None = None,
    ) -> Callable[[ResponderContext], Awaitable[ResponseSpec]]:
        middlewares: list[ResponderMiddleware] = [
            *self.responder_middlewares,
            *self._builtins.active(),
            RouterMiddleware(self.router),
            HandlerMiddleware(lambda: self._handler),
        ]

        def terminal(_: ResponderContext) -> ResponseSpec:
            return self._default_response

        return compose_responder(
            middlewares,
            terminal,
            capture_ctx=capture_ctx,
        )

    def _uses_static_response(self) -> bool:
        return (
            not self.responder_middlewares
            and not self._builtins.any_active
            and not self.router.has_routes
            and self._handler is None
        )

    async def _send_response(
        self,
        recording_writer: RecordingStreamWriter,
        request: RecordedHTTPRequest,
        response: ResponseSpec,
    ) -> SendResult:
        writer = recording_writer
        writer.start_response()

        if isinstance(response, ForwardProxyResponse):
            result = await response.forwarder.forward_and_relay(
                host=response.host,
                port=response.port,
                request_wire_bytes=response.request_wire_bytes,
                client_writer=writer,
                request_method=response.request_method,
                upstream_tls=response.upstream_tls,
            )
            wire_bytes = writer.bytes_sent
            if result is None:
                return await self._send_response(
                    recording_writer,
                    request,
                    HTTPResponse.text("Bad Gateway", status=502),
                )

            recorded = result.to_recorded_response(
                wire_raw_bytes=wire_bytes,
            )
            return SendResult(
                recorded=recorded,
                should_close=not response_allows_keep_alive(request, result),
            )

        if not isinstance(response, HTTPResponse):
            raise TypeError(
                f"Unhandled response spec: {type(response).__name__}"
            )

        should_close = await self._write_response(
            writer,
            response,
            request,
        )
        wire_bytes = writer.bytes_sent
        recorded = self._build_recorded_response(response, wire_bytes)
        return SendResult(recorded=recorded, should_close=should_close)

    def _build_sender(
        self,
        recording_writer: RecordingStreamWriter,
    ) -> Callable[[SenderContext, ResponseSpec], Awaitable[SendResult]]:
        middlewares = [*self.sender_middlewares]

        async def terminal(
            ctx: SenderContext,
            response: ResponseSpec,
        ) -> SendResult:
            return await self._send_response(
                recording_writer,
                ctx.request,
                response,
            )

        return compose_sender(middlewares, terminal)

    def _build_header(
        self,
    ) -> Callable[[HeaderContext], Awaitable[bool]]:
        middlewares: list[HeaderMiddleware] = []
        on_headers_received_handler = self._on_headers_received
        if on_headers_received_handler is not None:

            async def on_headers_received(
                ctx: HeaderContext,
                call_next: HeaderNext,
            ) -> bool:
                should_continue: bool = await maybe_await(
                    on_headers_received_handler(ctx.headers, ctx.send)
                )
                if not should_continue:
                    return False
                return await call_next()

            middlewares.append(on_headers_received)

        middlewares.extend(self.header_middlewares)

        async def terminal(_: HeaderContext) -> bool:
            return True

        return compose_headers(middlewares, terminal)

    async def _handle_request(
        self,
        *,
        request: RecordedHTTPRequest,
        request_timestamp: datetime,
        received_monotonic: float,
        state: dict[str, Any],
        recording_writer: RecordingStreamWriter,
    ) -> bool:
        exchange_recorded = False
        sender_request = request
        connection: ConnectionMeta | None = None
        sender = (
            self._build_sender(recording_writer)
            if self.sender_middlewares
            else None
        )
        try:
            if self._uses_static_response():
                response_spec: ResponseSpec = self._default_response
            else:

                def capture_responder_ctx(ctx: ResponderContext) -> None:
                    nonlocal sender_request
                    sender_request = ctx.request

                responder = self._build_responder(
                    capture_ctx=capture_responder_ctx
                )
                connection = ConnectionMeta(client=request.client)
                responder_ctx = ResponderContext(
                    request=request,
                    connection=connection,
                    services=self._services,
                    state=state,
                    received_monotonic=received_monotonic,
                )
                response_spec = await responder(responder_ctx)

            if sender is not None:
                if connection is None:
                    connection = ConnectionMeta(client=request.client)
                sender_ctx = SenderContext(
                    request=sender_request,
                    connection=connection,
                    services=self._services,
                    state=state,
                )
                send_result = await sender(sender_ctx, response_spec)
            else:
                send_result = await self._send_response(
                    recording_writer,
                    sender_request,
                    response_spec,
                )

            self._recorder.record_exchange(
                request=request,
                response=send_result.recorded,
                request_timestamp=request_timestamp,
            )
            exchange_recorded = True
            return send_result.should_close
        except Exception:
            if not exchange_recorded:
                self._recorder.record_exchange(
                    request=request,
                    response=None,
                    request_timestamp=request_timestamp,
                )
            raise

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._client_writers.add(writer)
        client = self._extract_client_info(writer)
        conn_recv, conn_sent = self._init_connection_tracking(client)
        recording_writer = RecordingStreamWriter(writer, conn_sent)
        try:
            while True:
                request_result = await self._read_request(
                    reader,
                    recording_writer,
                    client=client,
                    connection_wire=conn_recv,
                )
                if request_result is None:
                    break

                request, state = request_result
                (
                    request_timestamp,
                    received_monotonic,
                ) = self._recorder.record_request(request)
                should_close = await self._handle_request(
                    request=request,
                    request_timestamp=request_timestamp,
                    received_monotonic=received_monotonic,
                    state=state,
                    recording_writer=recording_writer,
                )

                if should_close:
                    break

        except Exception:
            LOG.exception("Error in AsyncHTTPTestServer handler")
        finally:
            try:
                recording_writer.close()
                await recording_writer.wait_closed()
            except Exception:
                LOG.debug("Failed to close client writer", exc_info=True)
            finally:
                self._client_writers.discard(writer)

    def _client_connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._closing:
            # Accepted so late in shutdown that aclose() already snapshotted
            # the writers to close. Close immediately so the client still gets
            # a FIN instead of waiting on a connection nobody owns.
            writer.close()
            return
        self._client_writers.add(writer)
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._client_tasks.add(task)

        def _release_client(done_task: asyncio.Task[None]) -> None:
            # A task canceled before its first step never enters
            # _handle_client, so its writer must be released here too.
            self._client_tasks.discard(done_task)
            self._client_writers.discard(writer)

        task.add_done_callback(_release_client)

    async def _read_request(
        self,
        reader: asyncio.StreamReader,
        writer: Writer,
        *,
        client: tuple[str, int] | None,
        connection_wire: BoundedByteBuffer | None = None,
    ) -> tuple[RecordedHTTPRequest, dict[str, Any]] | None:
        state: dict[str, Any] = {}
        parser = AsyncRequestParser()

        if not self.header_middlewares and self._on_headers_received is None:
            parsed, wire_bytes = await parser.parse(reader, connection_wire)
            if parsed is None:
                return None
            return self._build_request(parsed, wire_bytes, client), state

        parsed, header_wire, remaining = await parser.parse_headers(
            reader, connection_wire
        )
        if parsed is None:
            return None

        headers_msg = headers_to_headers(parsed.headers)
        partial = HTTPRequestHeaders(
            method=parsed.method,
            path=(
                parsed.url.decode("ascii", errors="replace")
                if parsed.url
                else None
            ),
            http_version=parsed.http_version,
            headers=headers_msg,
            wire_raw_bytes=header_wire,
        )

        async def send_response(response: HTTPResponse) -> None:
            await self._write_interim_response(writer, response)

        header_ctx = HeaderContext(
            headers=partial,
            connection=ConnectionMeta(client=client),
            services=self._services,
            send=send_response,
            state=state,
        )
        header_app = self._build_header()
        should_continue = await header_app(header_ctx)

        if not should_continue:
            return None

        parsed, wire_bytes = await parser.continue_parse_body(
            reader, remaining, connection_wire
        )
        if parsed is None:
            return None

        return self._build_request(parsed, wire_bytes, client), state

    def _build_request(
        self,
        parsed: ParsedRequest,
        wire_bytes: bytes,
        client: tuple[str, int] | None,
    ) -> RecordedHTTPRequest:
        """Build RecordedHTTPRequest from parsed data."""
        return RecordedHTTPRequest.from_parsed(
            parsed,
            wire_bytes,
            client=client,
        )

    def _build_recorded_response(
        self,
        response: HTTPResponse,
        wire_bytes: bytes,
    ) -> RecordedHTTPResponse:
        """Build RecordedHTTPResponse from response and wire bytes."""
        return RecordedHTTPResponse(
            response=HTTPResponse(
                status=response.status,
                headers=response.headers,
                body=self._normalize_body(response.body),
            ),
            reason=status_phrase(response.status),
            wire_raw_bytes=wire_bytes,
        )

    async def _write_interim_response(
        self,
        writer: Writer,
        response: HTTPResponse,
    ) -> None:
        """Write an interim response (e.g., 100 Continue) to the client."""
        reason = status_phrase(response.status, "UNKNOWN")

        status_line = f"HTTP/1.1 {response.status} {reason}\r\n"
        writer.write(status_line.encode("ascii"))

        is_informational = 100 <= response.status < 200
        body = b"" if is_informational else self._normalize_body(response.body)
        headers = self._build_response_headers(response, body, False)

        for name, value in headers:
            writer.write(serialize_header_line(name, value))

        writer.write(b"\r\n")
        if not is_informational and body:
            writer.write(body)
        await writer.drain()

    def _extract_client_info(self, writer: Writer) -> tuple[str, int] | None:
        """Extract client (host, port) from the writer's peername."""
        peer = writer.get_extra_info("peername")
        if isinstance(peer, tuple) and len(peer) >= 2:
            return (peer[0], peer[1])
        return None

    def _normalize_body(self, body_obj: bytes | str | None) -> bytes:
        """Normalize response body to bytes."""
        if body_obj is None:
            return b""
        if isinstance(body_obj, bytes):
            return body_obj
        return str(body_obj).encode("utf-8")

    def _build_response_headers(
        self,
        response: HTTPResponse,
        body: bytes,
        should_close: bool,
    ) -> tuple[HeaderItem, ...]:
        """Build the complete response header items, in order."""
        items = list(response.headers.items())
        header_names = {name.lower() for name, _ in items}

        if 100 <= response.status < 200 or response.status == 204:
            items = [
                (name, value)
                for name, value in items
                if name.lower() != "content-length"
            ]
        elif "content-length" not in header_names:
            items.append(("Content-Length", str(len(body))))

        if "connection" not in header_names and should_close:
            items.append(("Connection", "close"))

        return tuple(items)

    async def _write_response(
        self,
        writer: Writer,
        response: HTTPResponse,
        request: RecordedHTTPRequest,
    ) -> bool:
        body = self._normalize_body(response.body)
        body_allowed = request.method.upper() != "HEAD" and not (
            100 <= response.status < 200 or response.status in {204, 304}
        )
        wire_body = body if body_allowed else b""
        should_close = should_close_connection(
            request,
            response_headers=response.headers,
        )
        headers = self._build_response_headers(response, body, should_close)
        head = _serialize_response_head(
            response.status,
            headers,
        )

        if isinstance(self._transmission_strategy, ImmediateTransmission):
            writer.write(head + wire_body)
            await writer.drain()
            return should_close

        writer.write(head)
        await self._transmission_strategy.write_body(writer, wire_body)

        return should_close
