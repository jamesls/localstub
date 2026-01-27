from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import Message
from http import HTTPStatus
from typing import (
    Any,
    Awaitable,
    Callable,
    Hashable,
    Iterable,
    Optional,
    Protocol,
    cast,
)

import httpx

from localstub.forward import Forwarder, ForwardResult
from localstub.http.exchange import RecordedExchange
from localstub.http.request import (
    AsyncRequestParser,
    HTTPRequest,
    ParsedRequest,
)
from localstub.http.response import RecordedResponse
from localstub.http.uri import ParsedURI
from localstub.http.utils import headers_to_message
from localstub.throttle import (
    Clock,
    MonotonicClock,
    RequestThrottler,
    ThrottleDecision,
    TokenBucketThrottler,
)

LOG = logging.getLogger(__name__)


class TimestampProvider(Protocol):
    def now(self) -> datetime: ...


class SystemTimestampProvider:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass
class HTTPResponse:
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | str | None = b""

    @classmethod
    def json(
        cls,
        obj: Any,
        *,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> HTTPResponse:
        text = json.dumps(obj)
        body = text.encode("utf-8")
        base_headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        if headers:
            base_headers.update(headers)
        return cls(status=status, headers=base_headers, body=body)

    @classmethod
    def text(
        cls,
        text: str,
        *,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> HTTPResponse:
        body = text.encode("utf-8")
        base_headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": str(len(body)),
        }
        if headers:
            base_headers.update(headers)
        return cls(status=status, headers=base_headers, body=body)

    @classmethod
    def raw(
        cls,
        data: bytes,
        *,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> HTTPResponse:
        base_headers = {"Content-Length": str(len(data))}
        if headers:
            base_headers.update(headers)
        return cls(status=status, headers=base_headers, body=data)


Handler = Callable[[HTTPRequest], Awaitable[HTTPResponse] | HTTPResponse]

ThrottleKeyFunc = Callable[[HTTPRequest], Hashable]
ThrottleResponseFunc = Callable[[HTTPRequest, ThrottleDecision], HTTPResponse]
ThrottleResponse = HTTPResponse | ThrottleResponseFunc


def _default_throttle_key(_: HTTPRequest) -> str:
    return "global"


def _default_throttle_response(
    _: HTTPRequest,
    decision: ThrottleDecision,
) -> HTTPResponse:
    retry_after = max(1, math.ceil(decision.retry_after_seconds))
    return HTTPResponse.text(
        "Too Many Requests",
        status=429,
        headers={"Retry-After": str(retry_after)},
    )


@dataclass(frozen=True)
class ThrottleConfig:
    throttler: RequestThrottler
    response: ThrottleResponseFunc


@dataclass
class HTTPRequestHeaders:
    """Partial request available after headers are parsed, before body.

    This is provided to the on_headers_received callback, allowing inspection
    of request headers before the body is read. Useful for implementing
    HTTP 100-continue or early rejection based on headers.
    """

    method: str | None
    path: str | None
    http_version: str | None
    headers: Message
    wire_raw_bytes: bytes


# Callback to send a response to client during header processing
SendResponse = Callable[["HTTPResponse"], Awaitable[None]]

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
            pass


@dataclass(frozen=True)
class _PreparedResponse:
    response: HTTPResponse
    recorded_response: RecordedResponse
    should_close: bool


@dataclass
class ConnectionContext:
    """Per-connection tracking container.

    Minimal adapter that holds references to the connection-level wire buffers
    maintained by the server. Tests do not access this class; it is internal
    and used only to clarify responsibilities.
    """

    reader: asyncio.StreamReader
    writer: Writer
    client: tuple[str, int] | None
    raw_received_total: bytearray | None
    raw_sent_total: bytearray | None
    history: list[tuple[HTTPRequest, HTTPResponse]] = field(
        default_factory=list
    )


class Router:
    """Small router wrapper with optional method/path routes.

    MVP behavior: if no explicit route matches, falls back to the server's
    `handler` callable. If that is also absent, returns the configured
    default response.
    """

    def __init__(self) -> None:
        self._routes: dict[tuple[str, str], Handler] = {}

    def add_route(self, method: str, path: str, handler: Handler) -> None:
        self._routes[(method.upper(), path)] = handler

    async def resolve(
        self,
        request: HTTPRequest,
        fallback_handler: Handler | None,
        default_response: HTTPResponse,
    ) -> HTTPResponse:
        # Use effective_path for route matching to support both origin-form
        # ("/path") and absolute-form ("http://host/path") URIs
        key = (
            request.method.upper() if request.method else "",
            request.effective_path,
        )
        handler = self._routes.get(key, fallback_handler)

        if handler is None:
            return default_response

        result = handler(request)
        if isinstance(result, HTTPResponse):
            return result
        if inspect.isawaitable(result):
            return await cast(Awaitable[HTTPResponse], result)
        raise TypeError("Handler returned unsupported type")


class AsyncHTTPTestServer:
    """Small asyncio HTTP server used for testing SDK clients.

    Features:
      * exposes .url (e.g. "http://127.0.0.1:12345/")
      * records last_request (HTTPRequest) and a list of all requests
      * `wire_raw_bytes` contains the *exact* bytes received, including
        chunked / aws-chunked framing and trailers.
      * configurable static response, or plug in your own handler(request).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        handler: Handler | None = None,
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

        self._handler: Handler | None = handler
        self._default_response: HTTPResponse = (
            default_response or HTTPResponse.json({})
        )
        self._router = Router()
        self._on_headers_received = on_headers_received
        self._proxy_forwarder = proxy_forwarder
        self._raw_forwarder = raw_forwarder

        # Response sequence tracking
        self._response_sequence: list[HTTPResponse] = []
        self._response_sequence_index: int = 0

        # Transmission strategy for controlling how body bytes are sent
        self._transmission_strategy: TransmissionStrategy = (
            ImmediateTransmission()
        )

        self._throttle: ThrottleConfig | None = None

        self._clock: Clock = clock or MonotonicClock()
        self._request_timestamps: dict[int, float] = {}
        self._timestamp_provider = (
            timestamp_provider or SystemTimestampProvider()
        )

        self.last_request: HTTPRequest | None = None
        self.requests: list[HTTPRequest] = []
        self._request_queue: asyncio.Queue[HTTPRequest] = asyncio.Queue()
        self._response_queue: asyncio.Queue[RecordedResponse] = asyncio.Queue()

        self.last_response: RecordedResponse | None = None
        self.responses: list[RecordedResponse] = []

        self.last_exchange: RecordedExchange | None = None
        self.exchanges: list[RecordedExchange] = []

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
    def handler(self) -> Handler | None:
        return self._handler

    @handler.setter
    def handler(self, value: Handler | None) -> None:
        self._handler = value

    def add_route(self, method: str, path: str, handler: Handler) -> None:
        self._router.add_route(method, path, handler)

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
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """Configure a static JSON response returned for every request."""
        self._default_response = HTTPResponse.json(
            obj, status=status, headers=headers
        )
        # Clear any response sequence (last one wins)
        self._response_sequence = []
        self._response_sequence_index = 0

    def set_text_response(
        self,
        text: str,
        *,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._default_response = HTTPResponse.text(
            text,
            status=status,
            headers=headers,
        )
        # Clear any response sequence (last one wins)
        self._response_sequence = []
        self._response_sequence_index = 0

    def set_raw_response(
        self,
        data: bytes,
        *,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._default_response = HTTPResponse.raw(
            data,
            status=status,
            headers=headers,
        )
        # Clear any response sequence (last one wins)
        self._response_sequence = []
        self._response_sequence_index = 0

    def set_default_response(self, response: HTTPResponse) -> None:
        """Configure a static response returned for every request.

        Unlike set_json_response/set_text_response/set_raw_response, this
        accepts an already-constructed HTTPResponse object.

        Args:
            response: HTTPResponse object to return for all requests
        """
        self._default_response = response
        # Clear any response sequence (last one wins)
        self._response_sequence = []
        self._response_sequence_index = 0

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
        self._response_sequence = responses
        self._response_sequence_index = 0
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
        self._connection_raw_bytes_received.clear()
        self._connection_raw_bytes_sent.clear()
        # Reset response sequence index to allow reuse
        self._response_sequence_index = 0
        if self._throttle is not None:
            self._throttle.throttler.reset()

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
        self._throttle = ThrottleConfig(
            throttler=throttler,
            response=response_fn,
        )

    def clear_throttle(self) -> None:
        """Disable request-rate throttling."""
        self._throttle = None

    def _normalize_throttle_response(
        self,
        response: ThrottleResponse | None,
    ) -> ThrottleResponseFunc:
        if response is None:
            return _default_throttle_response
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
            self._handle_client,
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
        await self._server.wait_closed()
        self._server = None

    async def __aenter__(self) -> AsyncHTTPTestServer:
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

    async def _get_response(self, request: HTTPRequest) -> HTTPResponse:
        """Get response via sequence, router, handler, or default.

        Priority order:
        1. Response sequence (if set and not exhausted)
        2. Proxy forwarding (if proxy request and forwarder is set)
        3. Router with method/path matching
        4. Handler (if set)
        5. Default response
        """
        # Check sequence first - consumes next response if available
        if self._response_sequence and self._response_sequence_index < len(
            self._response_sequence
        ):
            response = self._response_sequence[self._response_sequence_index]
            self._response_sequence_index += 1
            return response

        # Forward proxy requests to upstream if forwarder is configured
        if request.is_proxy_request and self._proxy_forwarder is not None:
            return await self._forward_to_upstream(request)

        # Fall back to existing routing logic
        return await self._router.resolve(
            request, self._handler, self._default_response
        )

    async def _forward_to_upstream(
        self,
        request: HTTPRequest,
    ) -> HTTPResponse:
        """Forward a proxy request to the upstream server."""
        uri = request.target_uri
        if uri is None:
            return HTTPResponse(
                status=400,
                body=b"Bad Request: Not an absolute URI",
            )

        # Build the upstream URL (the full absolute URI from request.path)
        upstream_url = request.path or "/"

        # Build headers, excluding hop-by-hop headers
        hop_by_hop = {
            "connection",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "proxy-connection",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
        }
        headers: dict[str, str] = {}
        if request.headers:
            for name, value in request.headers.items():
                if name.lower() not in hop_by_hop:
                    headers[name] = value

        assert self._proxy_forwarder is not None
        try:
            upstream_response = await self._proxy_forwarder.request(
                method=request.method or "GET",
                url=upstream_url,
                headers=headers,
                content=request.body_bytes,
            )

            # Build response headers, excluding hop-by-hop
            response_headers: dict[str, str] = {}
            for name, value in upstream_response.headers.items():
                if name.lower() not in hop_by_hop:
                    response_headers[name] = value

            return HTTPResponse(
                status=upstream_response.status_code,
                headers=response_headers,
                body=upstream_response.content,
            )

        except httpx.RequestError as e:
            LOG.warning("Upstream request failed: %s", e)
            return HTTPResponse(
                status=502,
                body=f"Bad Gateway: {e}".encode(),
            )

    def _build_origin_form_request(
        self,
        request: HTTPRequest,
        uri: ParsedURI,
    ) -> bytes:
        """Convert absolute-form proxy request to origin-form for upstream.

        Converts a request like "GET http://example.com/path HTTP/1.1"
        to "GET /path HTTP/1.1" for sending to the upstream server.

        Args:
            request: The original HTTP request with absolute-form URI.
            uri: Parsed URI components from the request.

        Returns:
            Wire bytes for the origin-form request.
        """
        path = request.effective_path or "/"
        method = request.method or "GET"
        version_value = request.http_version or "1.1"
        if version_value.startswith("HTTP/"):
            version = version_value
        else:
            version = f"HTTP/{version_value}"

        lines = [f"{method} {path} {version}"]

        # Add headers, filtering proxy-specific hop-by-hop headers
        hop_by_hop = {
            "proxy-connection",
            "proxy-authenticate",
            "proxy-authorization",
        }
        connection_tokens = self._connection_tokens_from_headers(
            request.headers
        )
        remove_headers = hop_by_hop | {"connection"} | connection_tokens
        host_added = False
        if request.headers:
            for name, value in request.headers.items():
                name_lower = name.lower()
                if name_lower == "host":
                    # Ensure Host header matches target
                    port_suffix = ""
                    if uri.port and uri.port not in (80, 443):
                        port_suffix = f":{uri.port}"
                    lines.append(f"Host: {uri.host}{port_suffix}")
                    host_added = True
                elif name_lower in remove_headers:
                    continue
                else:
                    lines.append(f"{name}: {value}")

        if not host_added:
            port_suffix = ""
            if uri.port and uri.port not in (80, 443):
                port_suffix = f":{uri.port}"
            lines.append(f"Host: {uri.host}{port_suffix}")

        header_bytes = "\r\n".join(lines).encode("ascii") + b"\r\n\r\n"

        body_bytes = request.wire_body_bytes
        return header_bytes + body_bytes

    async def _forward_proxy_raw(
        self,
        request: HTTPRequest,
        writer: Writer,
    ) -> ForwardResult | None:
        """Forward a proxy request using raw sockets.

        Uses the Forwarder to send the request to upstream and relay
        the response directly to the client, preserving exact wire bytes
        including Transfer-Encoding.

        Args:
            request: The HTTP request to forward.
            writer: The writer to relay the response to.

        Returns:
            ForwardResult for recording, or None on failure.
        """
        if self._raw_forwarder is None:
            return None

        uri = request.target_uri
        if uri is None:
            return None

        request_wire = self._build_origin_form_request(request, uri)
        upstream_tls = uri.scheme == "https"

        return await self._raw_forwarder.forward_and_relay(
            host=uri.host,
            port=uri.port,
            request_wire_bytes=request_wire,
            client_writer=cast(asyncio.StreamWriter, writer),
            request_method=request.method,
            upstream_tls=upstream_tls,
        )

    async def _maybe_throttle_request(
        self,
        request: HTTPRequest,
        recording_writer: RecordingStreamWriter,
    ) -> _PreparedResponse | None:
        throttle = self._throttle
        if throttle is None:
            return None

        decision = throttle.throttler.check(request)
        if decision.allowed:
            return None

        response = throttle.response(request, decision)
        wire_offset = len(recording_writer.bytes_sent)
        should_close = await self._write_response(
            recording_writer,
            response,
            request,
        )
        wire_bytes = recording_writer.bytes_sent[wire_offset:]
        recorded = self._build_recorded_response(response, wire_bytes)
        return _PreparedResponse(
            response=response,
            recorded_response=recorded,
            should_close=should_close,
        )

    async def _maybe_forward_proxy_raw(
        self,
        request: HTTPRequest,
        recording_writer: RecordingStreamWriter,
    ) -> _PreparedResponse | None:
        if not request.is_proxy_request or self._raw_forwarder is None:
            return None

        wire_offset = len(recording_writer.bytes_sent)
        forward_result = await self._forward_proxy_raw(
            request,
            recording_writer,
        )
        if forward_result is None:
            return None

        response = HTTPResponse(
            status=forward_result.status,
            headers=dict(forward_result.headers.items()),
            body=forward_result.body,
        )
        wire_bytes = recording_writer.bytes_sent[wire_offset:]
        recorded = RecordedResponse(
            status=forward_result.status,
            reason=forward_result.reason,
            headers=forward_result.headers,
            body=forward_result.body.decode("utf-8", errors="replace"),
            wire_raw_bytes=wire_bytes,
        )
        return _PreparedResponse(
            response=response,
            recorded_response=recorded,
            should_close=self._should_close_connection(request, response),
        )

    async def _record_request(self, request: HTTPRequest) -> datetime:
        self.last_request = request
        self.requests.append(request)
        self._request_timestamps[id(request)] = self._clock.now()
        request_timestamp = self._timestamp_provider.now()
        await self._request_queue.put(request)
        return request_timestamp

    async def _handle_request(
        self,
        *,
        request: HTTPRequest,
        request_timestamp: datetime,
        recording_writer: RecordingStreamWriter,
        ctx: ConnectionContext,
    ) -> bool:
        exchange_recorded = False
        try:
            prepared = await self._maybe_throttle_request(
                request=request,
                recording_writer=recording_writer,
            )
            if prepared is None:
                prepared = await self._maybe_forward_proxy_raw(
                    request=request,
                    recording_writer=recording_writer,
                )

            if prepared is None:
                response = await self._get_response(request)
                wire_offset = len(recording_writer.bytes_sent)
                should_close = await self._write_response(
                    recording_writer, response, request
                )
                wire_bytes = recording_writer.bytes_sent[wire_offset:]
                recorded = self._build_recorded_response(response, wire_bytes)
                prepared = _PreparedResponse(
                    response=response,
                    recorded_response=recorded,
                    should_close=should_close,
                )

            response_timestamp = self._timestamp_provider.now()
            self._record_exchange(
                request=request,
                response=prepared.recorded_response,
                request_timestamp=request_timestamp,
                response_timestamp=response_timestamp,
            )
            exchange_recorded = True
            await self._response_queue.put(prepared.recorded_response)
            ctx.history.append((request, prepared.response))
            return prepared.should_close
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
        client = self._extract_client_info(writer)
        conn_recv, conn_sent = self._init_connection_tracking(client)
        recording_writer = RecordingStreamWriter(writer, conn_sent)
        ctx = ConnectionContext(
            reader=reader,
            writer=recording_writer,
            client=client,
            raw_received_total=conn_recv,
            raw_sent_total=conn_sent,
        )
        try:
            while True:
                request = await self._read_request(
                    reader, recording_writer, conn_recv
                )
                if request is None:
                    break
                request_timestamp = await self._record_request(request)
                should_close = await self._handle_request(
                    request=request,
                    request_timestamp=request_timestamp,
                    recording_writer=recording_writer,
                    ctx=ctx,
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
                pass

    async def _read_request(
        self,
        reader: asyncio.StreamReader,
        writer: Writer,
        connection_wire: bytearray | None = None,
    ) -> HTTPRequest | None:
        parser = AsyncRequestParser()

        # If no on_headers_received hook, use atomic parsing (unchanged)
        if self._on_headers_received is None:
            parsed, wire_bytes = await parser.parse(reader, connection_wire)
            if parsed is None:
                return None
            return self._build_request(parsed, wire_bytes, writer)

        # Two-phase parsing with on_headers_received hook
        # Phase 1: Parse headers only
        parsed, header_wire, remaining = await parser.parse_headers(
            reader, connection_wire
        )
        if parsed is None:
            return None

        # Build partial request for the hook
        headers_msg = headers_to_message(parsed.headers)
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

        # Create the send callback
        async def send_response(response: HTTPResponse) -> None:
            await self._write_interim_response(writer, response)

        # Call the hook
        result = self._on_headers_received(partial, send_response)
        if inspect.isawaitable(result):
            should_continue = await result
        else:
            should_continue = cast(bool, result)

        if not should_continue:
            return None

        # Phase 2: Parse body
        parsed, wire_bytes = await parser.continue_parse_body(
            reader, remaining, connection_wire
        )
        if parsed is None:
            return None

        return self._build_request(parsed, wire_bytes, writer)

    def _build_request(
        self,
        parsed: ParsedRequest,
        wire_bytes: bytes,
        writer: Writer,
    ) -> HTTPRequest:
        """Build HTTPRequest from parsed data."""
        return HTTPRequest.from_parsed(parsed, wire_bytes, writer=writer)

    def _build_recorded_response(
        self,
        response: HTTPResponse,
        wire_bytes: bytes,
    ) -> RecordedResponse:
        """Build RecordedResponse from response and wire bytes."""
        try:
            reason = HTTPStatus(response.status).phrase
        except ValueError:
            reason = None

        body = self._normalize_body(response.body)
        body_text = body.decode("utf-8", errors="replace") if body else None

        headers = headers_to_message(
            [(k.encode(), v.encode()) for k, v in response.headers.items()]
            if response.headers
            else []
        )

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
        try:
            reason = HTTPStatus(response.status).phrase
        except ValueError:
            reason = "UNKNOWN"

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

    def _parse_connection_tokens(self, value: str) -> set[str]:
        tokens: set[str] = set()
        for raw_token in value.split(","):
            token = raw_token.strip().lower()
            if token:
                tokens.add(token)
        return tokens

    def _connection_tokens_from_headers(
        self,
        headers: Message | None,
    ) -> set[str]:
        if headers is None:
            return set()
        tokens: set[str] = set()
        for value in headers.get_all("Connection", []):
            tokens.update(self._parse_connection_tokens(value))
        return tokens

    def _connection_tokens_from_response(
        self,
        response: HTTPResponse,
    ) -> set[str]:
        for name, value in response.headers.items():
            if name.lower() == "connection":
                return self._parse_connection_tokens(value)
        return set()

    def _is_http10(self, request: HTTPRequest) -> bool:
        return request.http_version == "1.0"

    def _is_http11(self, request: HTTPRequest) -> bool:
        return request.http_version == "1.1"

    def _should_close_connection(
        self,
        request: HTTPRequest,
        response: HTTPResponse,
    ) -> bool:
        """Return True if the server should close after this response."""
        request_tokens = self._connection_tokens_from_headers(request.headers)
        response_tokens = self._connection_tokens_from_response(response)

        if "close" in response_tokens:
            return True

        if self._is_http11(request):
            return "close" in request_tokens

        if self._is_http10(request):
            return "keep-alive" not in request_tokens

        return True

    def _build_response_headers(
        self,
        response: HTTPResponse,
        body: bytes,
        should_close: bool,
    ) -> dict[str, str]:
        """Build complete response headers dict."""
        headers = dict(response.headers) if response.headers else {}
        header_names = {k.lower() for k in headers}

        if 100 <= response.status < 200:
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
        try:
            reason = HTTPStatus(response.status).phrase
        except ValueError:
            reason = "UNKNOWN"

        status_line = f"HTTP/1.1 {response.status} {reason}\r\n"
        writer.write(status_line.encode("ascii"))

        body = self._normalize_body(response.body)
        should_close = self._should_close_connection(request, response)
        headers = self._build_response_headers(response, body, should_close)

        for name, value in headers.items():
            header_line = f"{name}: {value}\r\n".encode("ascii")
            writer.write(header_line)

        writer.write(b"\r\n")
        await self._transmission_strategy.write_body(writer, body)

        return should_close
