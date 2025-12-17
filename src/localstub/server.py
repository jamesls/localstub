from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from email.message import Message
from http import HTTPStatus
from typing import Any, Awaitable, Callable, Iterable, Optional, Protocol, cast

from localstub.http.request import (
    AsyncRequestParser,
    HTTPRequest,
    ParsedRequest,
)
from localstub.http.utils import headers_to_message

LOG = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# Transmission strategies
# ---------------------------------------------------------------------------


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


class HTTPProtocol:
    """Thin protocol facade that delegates to existing helpers.

    This allows us to separate parsing/formatting concerns from the server
    loop without changing the tested public API or behavior.
    """

    def __init__(self, server: "AsyncHTTPTestServer") -> None:  # noqa: F821
        self._server = server

    async def parse_request(
        self,
        reader: asyncio.StreamReader,
        writer: Writer,
        conn_recv: bytearray | None,
    ) -> HTTPRequest | None:
        return await self._server._read_request(reader, writer, conn_recv)

    async def send_response(
        self,
        writer: Writer,
        response: HTTPResponse,
        request: HTTPRequest,
    ) -> bool:
        return await self._server._write_response(writer, response, request)


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
        key = (
            request.method.upper() if request.method else "",
            request.path or "/",
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
        chunked framing and trailers.
      * configurable static response, or plug in your own handler(request).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        handler: Handler | None = None,
        default_response: HTTPResponse | None = None,
        on_headers_received: OnHeadersReceived | None = None,
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

        # Response sequence tracking
        self._response_sequence: list[HTTPResponse] = []
        self._response_sequence_index: int = 0

        # Transmission strategy for controlling how body bytes are sent
        self._transmission_strategy: TransmissionStrategy = (
            ImmediateTransmission()
        )

        self.last_request: HTTPRequest | None = None
        self.requests: list[HTTPRequest] = []
        self._request_queue: asyncio.Queue[HTTPRequest] = asyncio.Queue()

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

    # Optional convenience for method/path handlers; tests don't use this.
    def add_route(self, method: str, path: str, handler: Handler) -> None:
        self._router.add_route(method, path, handler)

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
        self._request_queue = asyncio.Queue()
        self._connection_raw_bytes_received.clear()
        self._connection_raw_bytes_sent.clear()
        # Reset response sequence index to allow reuse
        self._response_sequence_index = 0

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
        2. Router with method/path matching
        3. Handler (if set)
        4. Default response
        """
        # Check sequence first - consumes next response if available
        if self._response_sequence and self._response_sequence_index < len(
            self._response_sequence
        ):
            response = self._response_sequence[self._response_sequence_index]
            self._response_sequence_index += 1
            return response

        # Fall back to existing routing logic
        return await self._router.resolve(
            request, self._handler, self._default_response
        )

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        client = self._extract_client_info(writer)
        conn_recv, conn_sent = self._init_connection_tracking(client)
        recording_writer = RecordingStreamWriter(writer, conn_sent)
        protocol = HTTPProtocol(self)
        ctx = ConnectionContext(
            reader=reader,
            writer=recording_writer,
            client=client,
            raw_received_total=conn_recv,
            raw_sent_total=conn_sent,
        )

        try:
            while True:
                request = await protocol.parse_request(
                    reader, recording_writer, conn_recv
                )
                if request is None:
                    break

                self.last_request = request
                self.requests.append(request)
                await self._request_queue.put(request)

                response = await self._get_response(request)
                should_close = await protocol.send_response(
                    recording_writer, response, request
                )

                # Keep a lightweight per-connection history for debugging
                ctx.history.append((request, response))

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
        headers = headers_to_message(parsed.headers)
        body_text = (
            parsed.body.decode("utf-8", errors="replace")
            if parsed.body
            else None
        )
        client = self._extract_client_info(writer)
        return HTTPRequest(
            method=parsed.method,
            path=(
                parsed.url.decode("ascii", errors="replace")
                if parsed.url
                else None
            ),
            http_version=parsed.http_version,
            headers=headers,
            body=body_text,
            wire_raw_bytes=wire_bytes,
            client=client,
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

    def _should_close_connection(self, request: HTTPRequest) -> bool:
        """Check if client requested connection close."""
        if not request.headers:
            return False
        client_conn = request.headers.get("Connection", "").lower()
        return "close" in client_conn

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
        should_close = self._should_close_connection(request)
        headers = self._build_response_headers(response, body, should_close)

        for name, value in headers.items():
            header_line = f"{name}: {value}\r\n".encode("ascii")
            writer.write(header_line)

        writer.write(b"\r\n")
        await self._transmission_strategy.write_body(writer, body)

        return should_close
