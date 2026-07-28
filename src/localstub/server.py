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
    cast,
)

import httpx

from localstub.forward import Forwarder
from localstub.http.connection import should_close_connection
from localstub.http.exchange import RecordedExchange
from localstub.http.request import (
    AsyncRequestParser,
    HTTPRequest,
    HTTPRequestHeaders,
    ParsedRequest,
)
from localstub.http.response import RecordedResponse
from localstub.http.responsespec import HTTPResponse
from localstub.http.utils import (
    headers_to_headers,
    maybe_await,
    message_from_items,
    status_phrase,
)
from localstub.middleware import (
    ConnectionMeta,
    ForwardProxyResponse,
    HeaderContext,
    HeaderMiddleware,
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
    HandlerMiddleware,
    HttpxForwardProxyMiddleware,
    RawForwardProxyMiddleware,
    ResponseSequenceMiddleware,
    RouterMiddleware,
    ThrottleMiddleware,
    ThrottleResponseFunc,
    default_throttle_response,
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


def _default_throttle_key(_: HTTPRequest) -> str:
    return "global"


@lru_cache(maxsize=128)
def _serialize_response_head(
    status: int,
    headers: tuple[tuple[str, str], ...],
) -> bytes:
    reason = status_phrase(status, "UNKNOWN")
    head = bytearray(f"HTTP/1.1 {status} {reason}\r\n".encode("ascii"))
    for name, value in headers:
        head.extend(f"{name}: {value}\r\n".encode("ascii"))
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

    async def drain(self) -> Any: ...

    def write_eof(self) -> None: ...

    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...

    def is_closing(self) -> bool: ...

    def get_extra_info(self, name: str, default: Any | None = None) -> Any: ...


class RecordingStreamWriter:
    """Wrapper that records all bytes written to the underlying writer."""

    def __init__(
        self,
        writer: asyncio.StreamWriter,
        sent_buffer: bytearray | None = None,
    ) -> None:
        self._writer = writer
        self._sent_buffer = sent_buffer
        self._recorded = bytearray()

    @property
    def bytes_sent(self) -> bytes:
        """Return all bytes written through this recorder."""
        return bytes(self._recorded)

    @property
    def bytes_sent_len(self) -> int:
        """Length of recorded bytes without copying."""
        return len(self._recorded)

    def bytes_sent_since(self, offset: int) -> bytes:
        """Return bytes written since *offset* without a full copy."""
        return bytes(self._recorded[offset:])

    def write(self, data: bytes) -> None:
        self._recorded.extend(data)
        if self._sent_buffer is not None:
            self._sent_buffer.extend(data)
        self._writer.write(data)

    def writelines(self, data: Iterable[bytes]) -> None:
        for chunk in data:
            self.write(chunk)

    async def drain(self) -> Any:
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

    def __init__(self, chunk_size: int, delay: float) -> None:
        """Initialize throttled transmission.

        Args:
            chunk_size: Number of bytes to send in each chunk
            delay: Seconds to wait between chunks
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size
        self.delay = delay

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
                await asyncio.sleep(self.delay)


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
      * records last_request (HTTPRequest) and a list of all requests
      * `wire_raw_bytes` contains the *exact* bytes received, including
        chunked / aws-chunked framing and trailers.
      * configurable static response, or plug in responder middleware.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        handler: ResponderHandler | None = None,
        default_response: HTTPResponse | None = None,
        on_headers_received: OnHeadersReceived | None = None,
        proxy_forwarder: httpx.AsyncClient | None = None,
        raw_forwarder: Forwarder | None = None,
        clock: Clock | None = None,
        timestamp_provider: TimestampProvider | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._server: asyncio.base_events.Server | None = None
        self._client_writers: set[asyncio.StreamWriter] = set()
        self._client_tasks: set[asyncio.Task[None]] = set()

        self._handler: ResponderHandler | None = handler
        self._default_response: HTTPResponse = (
            default_response or HTTPResponse.json({})
        )
        self.router = Router()
        self._on_headers_received = on_headers_received
        self._proxy_forwarder = proxy_forwarder
        self._raw_forwarder = raw_forwarder

        self._throttle_middleware: ThrottleMiddleware | None = None
        self._response_sequence_middleware: (
            ResponseSequenceMiddleware | None
        ) = None
        self._raw_forward_proxy_middleware: (
            RawForwardProxyMiddleware | None
        ) = (
            RawForwardProxyMiddleware(raw_forwarder)
            if raw_forwarder is not None
            else None
        )
        self._httpx_forward_proxy_middleware: (
            HttpxForwardProxyMiddleware | None
        ) = (
            HttpxForwardProxyMiddleware(proxy_forwarder)
            if proxy_forwarder is not None
            else None
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

        self._request_timestamps: dict[int, float] = {}

        self.last_request: HTTPRequest | None = None
        self.requests: list[HTTPRequest] = []
        self._request_queue: asyncio.Queue[HTTPRequest] = asyncio.Queue()
        self._response_queue: asyncio.Queue[RecordedResponse] = asyncio.Queue()

        self.last_response: RecordedResponse | None = None
        self.responses: list[RecordedResponse] = []

        self.last_exchange: RecordedExchange | None = None
        self.exchanges: list[RecordedExchange] = []
        self._exchange_queue: asyncio.Queue[RecordedExchange] = asyncio.Queue()

        # Connection-level raw bytes tracking (keyed by client address)
        self._connection_raw_bytes_received: dict[
            tuple[str, int], bytearray
        ] = {}
        self._connection_raw_bytes_sent: dict[tuple[str, int], bytearray] = {}

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
        self._default_response = response
        self._response_sequence_middleware = None

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
        headers: dict[str, str] | None = None,
    ) -> None:
        """Configure a static JSON response returned for every request."""
        self._default_response = HTTPResponse.json(
            obj, status=status, headers=headers
        )
        self._response_sequence_middleware = None

    def set_text_response(
        self,
        text: str,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._default_response = HTTPResponse.text(
            text,
            status=status,
            headers=headers,
        )
        self._response_sequence_middleware = None

    def set_raw_response(
        self,
        data: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self._default_response = HTTPResponse.raw(
            data,
            status=status,
            headers=headers,
        )
        self._response_sequence_middleware = None

    def set_default_response(self, response: HTTPResponse) -> None:
        """Configure a static response returned for every request.

        Unlike set_json_response/set_text_response/set_raw_response, this
        accepts an already-constructed HTTPResponse object.

        Args:
            response: HTTPResponse object to return for all requests
        """
        self._default_response = response
        self._response_sequence_middleware = None

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
        self._response_sequence_middleware = ResponseSequenceMiddleware(
            responses
        )
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
        """Get all raw bytes received from a specific client connection.

        Args:
            client: Tuple of (host, port) identifying the client connection

        Returns:
            All bytes received from this client, or None if no data recorded
        """
        buf = self._connection_raw_bytes_received.get(client)
        return bytes(buf) if buf is not None else None

    def get_connection_bytes_sent(
        self, client: tuple[str, int]
    ) -> bytes | None:
        """Get all raw bytes sent to a specific client connection.

        Args:
            client: Tuple of (host, port) identifying the client connection

        Returns:
            All bytes sent to this client, or None if no data recorded
        """
        buf = self._connection_raw_bytes_sent.get(client)
        return bytes(buf) if buf is not None else None

    def get_request_timestamp(self, request: HTTPRequest) -> float:
        """Get the reception timestamp for a request.

        Args:
            request: The HTTPRequest object to look up.

        Returns:
            Monotonic timestamp when the request was received.

        Raises:
            ValueError: If the request is not found.
        """
        try:
            return self._request_timestamps[id(request)]
        except KeyError:
            raise ValueError(
                "Request not found in recorded requests"
            ) from None

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
        self.last_request = None
        self.requests = []
        self._request_timestamps = {}
        self._request_queue = asyncio.Queue()
        self._response_queue = asyncio.Queue()
        self.last_response = None
        self.responses = []
        self.last_exchange = None
        self.exchanges = []
        self._exchange_queue = asyncio.Queue()
        self._connection_raw_bytes_received.clear()
        self._connection_raw_bytes_sent.clear()
        if self._response_sequence_middleware is not None:
            self._response_sequence_middleware.reset()
        if self._throttle_middleware is not None:
            self._throttle_middleware.reset()

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
        self._throttle_middleware = ThrottleMiddleware(
            throttler=throttler,
            response=response_fn,
        )

    def clear_throttle(self) -> None:
        """Disable request-rate throttling."""
        self._throttle_middleware = None

    def _normalize_throttle_response(
        self,
        response: ThrottleResponse | None,
    ) -> ThrottleResponseFunc:
        if response is None:
            return default_throttle_response
        if isinstance(response, HTTPResponse):

            def static(
                _: HTTPRequest,
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

        self._server = await asyncio.start_server(
            self._client_connected,
            self._host,
            self._port,
        )
        assert self._server.sockets
        sockname = self._server.sockets[0].getsockname()
        self.host, self.port = sockname[0], sockname[1]

    async def aclose(self) -> None:
        if self._server is None:
            return
        self._server.close()
        # Let callbacks for connections accepted before close register their
        # writers and tasks before taking the shutdown snapshot.
        await asyncio.sleep(0)
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
        await self._server.wait_closed()
        self._client_tasks.clear()
        self._server = None

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def next_request(self, timeout: float | None = None) -> HTTPRequest:
        """Await and return the next request that hits this server."""
        if timeout is None:
            req = await self._request_queue.get()
        else:
            req = await asyncio.wait_for(
                self._request_queue.get(), timeout=timeout
            )
        self.last_request = req
        return req

    async def next_response(
        self, timeout: float | None = None
    ) -> RecordedResponse:
        """Await and return the next response sent by this server."""
        if timeout is None:
            response = await self._response_queue.get()
        else:
            response = await asyncio.wait_for(
                self._response_queue.get(), timeout=timeout
            )
        self.last_response = response
        return response

    async def next_exchange(
        self, timeout: float | None = None
    ) -> RecordedExchange:
        """Await and return the next completed request/response exchange."""
        if timeout is None:
            return await self._exchange_queue.get()
        return await asyncio.wait_for(
            self._exchange_queue.get(), timeout=timeout
        )

    def _record_exchange(
        self,
        *,
        request: HTTPRequest,
        response: RecordedResponse | None,
        request_timestamp: datetime,
        response_timestamp: datetime | None,
    ) -> None:
        exchange = RecordedExchange(
            request=request,
            response=response,
            request_timestamp=request_timestamp,
            response_timestamp=response_timestamp,
        )
        self.exchanges.append(exchange)
        self.last_exchange = exchange
        self._exchange_queue.put_nowait(exchange)
        if response is not None:
            self.responses.append(response)
            self.last_response = response

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
    ) -> tuple[bytearray | None, bytearray | None]:
        """Initialize connection-level byte tracking for a client."""
        if client is None:
            return None, None

        if client not in self._connection_raw_bytes_received:
            self._connection_raw_bytes_received[client] = bytearray()
        if client not in self._connection_raw_bytes_sent:
            self._connection_raw_bytes_sent[client] = bytearray()

        return (
            self._connection_raw_bytes_received[client],
            self._connection_raw_bytes_sent[client],
        )

    def _build_responder(
        self,
        *,
        capture_ctx: Callable[[ResponderContext], None] | None = None,
    ) -> Callable[[ResponderContext], Awaitable[ResponseSpec]]:
        builtins: list[ResponderMiddleware] = []
        if self._throttle_middleware is not None:
            builtins.append(self._throttle_middleware)
        if self._response_sequence_middleware is not None:
            builtins.append(self._response_sequence_middleware)
        if self._raw_forward_proxy_middleware is not None:
            builtins.append(self._raw_forward_proxy_middleware)
        if self._httpx_forward_proxy_middleware is not None:
            builtins.append(self._httpx_forward_proxy_middleware)
        builtins.append(RouterMiddleware(self.router))
        builtins.append(HandlerMiddleware(lambda: self._handler))

        middlewares = [*self.responder_middlewares, *builtins]

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
            and self._throttle_middleware is None
            and self._response_sequence_middleware is None
            and self._raw_forward_proxy_middleware is None
            and self._httpx_forward_proxy_middleware is None
            and not self.router.has_routes
            and self._handler is None
        )

    async def _send_response(
        self,
        recording_writer: RecordingStreamWriter,
        request: HTTPRequest,
        response: ResponseSpec,
    ) -> SendResult:
        writer = recording_writer

        if isinstance(response, ForwardProxyResponse):
            wire_offset = writer.bytes_sent_len
            result = await response.forwarder.forward_and_relay(
                host=response.host,
                port=response.port,
                request_wire_bytes=response.request_wire_bytes,
                client_writer=cast(Any, writer),
                request_method=response.request_method,
                upstream_tls=response.upstream_tls,
            )
            wire_bytes = writer.bytes_sent_since(wire_offset)
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
                should_close=(
                    result.is_eof_delimited
                    or should_close_connection(
                        request,
                        response_headers=result.headers,
                    )
                ),
            )

        if not isinstance(response, HTTPResponse):
            raise TypeError(
                f"Unhandled response spec: {type(response).__name__}"
            )

        wire_offset = writer.bytes_sent_len
        should_close = await self._write_response(
            writer,
            response,
            request,
        )
        wire_bytes = writer.bytes_sent_since(wire_offset)
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
                call_next: Any,
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

    def _record_request(
        self,
        request: HTTPRequest,
    ) -> tuple[datetime, float]:
        received_monotonic = self._clock.now()
        self.last_request = request
        self.requests.append(request)
        self._request_timestamps[id(request)] = received_monotonic
        request_timestamp = self._timestamp_provider.now()
        self._request_queue.put_nowait(request)
        return request_timestamp, received_monotonic

    async def _handle_request(
        self,
        *,
        request: HTTPRequest,
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

            response_timestamp = self._timestamp_provider.now()
            self._record_exchange(
                request=request,
                response=send_result.recorded,
                request_timestamp=request_timestamp,
                response_timestamp=response_timestamp,
            )
            exchange_recorded = True
            self._response_queue.put_nowait(send_result.recorded)
            return send_result.should_close
        except Exception:
            if not exchange_recorded:
                self._record_exchange(
                    request=request,
                    response=None,
                    request_timestamp=request_timestamp,
                    response_timestamp=None,
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
                ) = self._record_request(request)
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
        connection_wire: bytearray | None = None,
    ) -> tuple[HTTPRequest, dict[str, Any]] | None:
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
    ) -> HTTPRequest:
        """Build HTTPRequest from parsed data."""
        return HTTPRequest.from_parsed(
            parsed,
            wire_bytes,
            client=client,
        )

    def _build_recorded_response(
        self,
        response: HTTPResponse,
        wire_bytes: bytes,
    ) -> RecordedResponse:
        """Build RecordedResponse from response and wire bytes."""
        reason = status_phrase(response.status)

        body = self._normalize_body(response.body)
        body_text = body.decode("utf-8", errors="replace") if body else None

        headers = message_from_items(response.headers.items())

        return RecordedResponse(
            status=response.status,
            reason=reason,
            headers=headers,
            body=body_text,
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

        for name, value in headers.items():
            header_line = f"{name}: {value}\r\n".encode("ascii")
            writer.write(header_line)

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
    ) -> dict[str, str]:
        """Build complete response headers dict."""
        headers = dict(response.headers) if response.headers else {}
        header_names = {k.lower() for k in headers}

        if 100 <= response.status < 200 or response.status == 204:
            headers = {
                name: value
                for name, value in headers.items()
                if name.lower() != "content-length"
            }
            header_names = {k.lower() for k in headers}
        elif "content-length" not in header_names:
            headers["Content-Length"] = str(len(body))

        if "connection" not in header_names and should_close:
            headers["Connection"] = "close"

        return headers

    async def _write_response(
        self,
        writer: Writer,
        response: HTTPResponse,
        request: HTTPRequest,
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
            tuple(headers.items()),
        )

        if isinstance(self._transmission_strategy, ImmediateTransmission):
            writer.write(head + wire_body)
            await writer.drain()
            return should_close

        writer.write(head)
        await self._transmission_strategy.write_body(writer, wire_body)

        return should_close
