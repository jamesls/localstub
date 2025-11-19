from __future__ import annotations

import asyncio
import inspect
import json
import logging
from dataclasses import dataclass, field
from email.message import Message
from http import HTTPStatus
from typing import Any, Awaitable, Callable, Optional

LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Recorded request
# ---------------------------------------------------------------------------


@dataclass
class HTTPRequest:
    """Snapshot of a single HTTP request.

    `body` is decoded as UTF-8 (like the original code).
    `wire_raw_bytes` is *exactly* what came off the wire, including:
      - request line
      - headers
      - the blank line
      - body bytes (including chunk framing for chunked requests)
    """

    method: str | None = None
    path: str | None = None
    http_version: str | None = None
    headers: Message | None = None
    body: str | None = None
    wire_raw_bytes: bytes | None = None
    client: tuple[str, int] | None = None

    @property
    def json_body(self) -> Any:
        if self.body is None or self.body == "":
            return None
        return json.loads(self.body)


def _parse_headers(header_lines: list[bytes]) -> Message:
    """Parse raw header lines into an email.message.Message (HTTP-style)."""
    msg = Message()
    current_name: str | None = None
    current_value_parts: list[str] = []

    for raw in header_lines:
        line = raw.decode("iso-8859-1").rstrip("\r\n")
        if not line and current_name is None:
            continue

        # Continuation line
        if line.startswith((" ", "\t")) and current_name is not None:
            current_value_parts.append(line.strip())
            continue

        # Finish previous header
        if current_name is not None:
            msg[current_name] = " ".join(current_value_parts)
            current_name = None
            current_value_parts = []

        if ":" not in line:
            # Malformed line, ignore
            continue

        name, value = line.split(":", 1)
        current_name = name.strip()
        current_value_parts = [value.strip()]

    if current_name is not None:
        msg[current_name] = " ".join(current_value_parts)

    return msg


# ---------------------------------------------------------------------------
# Stubbed response
# ---------------------------------------------------------------------------


@dataclass
class StubResponse:
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
    ) -> StubResponse:
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
    ) -> StubResponse:
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
    ) -> StubResponse:
        base_headers = {"Content-Length": str(len(data))}
        if headers:
            base_headers.update(headers)
        return cls(status=status, headers=base_headers, body=data)


Handler = Callable[[HTTPRequest], Awaitable[StubResponse] | StubResponse]


# ---------------------------------------------------------------------------
# Async HTTP server
# ---------------------------------------------------------------------------


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
        default_response: StubResponse | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._server: asyncio.base_events.Server | None = None

        self._handler: Handler | None = handler
        self._default_response: StubResponse = (
            default_response or StubResponse.json({})
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

    def set_json_response(
        self,
        obj: Any,
        *,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        """Configure a static JSON response returned for every request."""
        self._default_response = StubResponse.json(
            obj, status=status, headers=headers
        )

    def set_text_response(
        self,
        text: str,
        *,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._default_response = StubResponse.text(
            text,
            status=status,
            headers=headers,
        )

    def set_raw_response(
        self,
        data: bytes,
        *,
        status: int = 200,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        self._default_response = StubResponse.raw(
            data,
            status=status,
            headers=headers,
        )

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    async def _get_response(self, request: HTTPRequest) -> StubResponse:
        """Get response for a request, using handler or default."""
        if self._handler is None:
            return self._default_response

        result = self._handler(request)
        if inspect.isawaitable(result):
            return await result  # type: ignore[return-value]
        return result

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        client = self._extract_client_info(writer)
        conn_recv, conn_sent = self._init_connection_tracking(client)

        try:
            while True:
                request = await self._read_request(reader, writer, conn_recv)
                if request is None:
                    break

                self.last_request = request
                self.requests.append(request)
                await self._request_queue.put(request)

                response = await self._get_response(request)
                should_close = await self._write_response(
                    writer, response, request, conn_sent
                )
                if should_close:
                    break

        except Exception:
            LOG.exception("Error in AsyncHTTPTestServer handler")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _read_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        connection_wire: bytearray | None = None,
    ) -> HTTPRequest | None:
        wire = bytearray()
        header_lines: list[bytes] = []

        # --- Request line ---
        req_line_parts = await self._read_request_line(
            reader, wire, connection_wire
        )
        if req_line_parts is None:
            return None
        method, path, version = req_line_parts

        # --- Headers ---
        headers = await self._read_headers(
            reader, wire, header_lines, connection_wire
        )
        if headers is None:
            return None

        # --- Body ---
        body_text = await self._read_body(
            reader, wire, headers, connection_wire
        )

        # --- Client info ---
        client = self._extract_client_info(writer)

        return HTTPRequest(
            method=method,
            path=path,
            http_version=version,
            headers=headers,
            body=body_text,
            wire_raw_bytes=bytes(wire),
            client=client,
        )

    async def _read_request_line(
        self,
        reader: asyncio.StreamReader,
        wire: bytearray,
        connection_wire: bytearray | None = None,
    ) -> tuple[str, str, str] | None:
        """Read and parse the HTTP request line."""
        line = await reader.readline()
        if not line:
            return None
        wire.extend(line)
        if connection_wire is not None:
            connection_wire.extend(line)
        try:
            req_line = line.decode("ascii", errors="replace").rstrip("\r\n")
            method, path, version = req_line.split(" ", 2)
            return (method, path, version)
        except ValueError:
            LOG.debug("Malformed request line: %r", line)
            return None

    async def _read_headers(
        self,
        reader: asyncio.StreamReader,
        wire: bytearray,
        header_lines: list[bytes],
        connection_wire: bytearray | None = None,
    ) -> Message | None:
        """Read HTTP headers until blank line."""
        while True:
            line = await reader.readline()
            if not line:
                # EOF while reading headers
                return None
            wire.extend(line)
            if connection_wire is not None:
                connection_wire.extend(line)
            if line in (b"\r\n", b"\n"):
                break
            header_lines.append(line)
        return _parse_headers(header_lines)

    async def _read_body(
        self,
        reader: asyncio.StreamReader,
        wire: bytearray,
        headers: Message,
        connection_wire: bytearray | None = None,
    ) -> str | None:
        """Read request body based on Transfer-Encoding or Content-Length."""
        body_bytes: bytes | None = None
        transfer_encoding = headers.get("Transfer-Encoding")
        if transfer_encoding and "chunked" in transfer_encoding.lower():
            body_bytes = await self._read_chunked_body(
                reader, wire, connection_wire
            )
        else:
            body_bytes = await self._read_content_length_body(
                reader, wire, headers, connection_wire
            )

        if body_bytes is None:
            return None
        return body_bytes.decode("utf-8", errors="replace")

    async def _read_content_length_body(
        self,
        reader: asyncio.StreamReader,
        wire: bytearray,
        headers: Message,
        connection_wire: bytearray | None = None,
    ) -> bytes | None:
        """Read body based on Content-Length header."""
        content_length = headers.get("Content-Length")
        if content_length is None:
            return None
        try:
            length = int(content_length)
        except ValueError:
            length = 0
        if length:
            chunk = await reader.readexactly(length)
            wire.extend(chunk)
            if connection_wire is not None:
                connection_wire.extend(chunk)
            return chunk
        return None

    def _extract_client_info(
        self, writer: asyncio.StreamWriter
    ) -> tuple[str, int] | None:
        """Extract client (host, port) from the writer's peername."""
        peer = writer.get_extra_info("peername")
        if isinstance(peer, tuple) and len(peer) >= 2:
            return (peer[0], peer[1])
        return None

    def _extend_wire_buffers(
        self,
        data: bytes,
        wire: bytearray,
        connection_wire: bytearray | None,
    ) -> None:
        """Extend both per-request and connection-level wire buffers."""
        wire.extend(data)
        if connection_wire is not None:
            connection_wire.extend(data)

    def _parse_chunk_size(self, header: str) -> int | None:
        """Parse chunk size from header, handling aws-chunked format."""
        size_str = header.split(";", 1)[0] if ";" in header else header
        try:
            return int(size_str, 16)
        except ValueError:
            LOG.debug("Invalid chunk size header: %r", header)
            return None

    async def _read_chunk_trailers(
        self,
        reader: asyncio.StreamReader,
        wire: bytearray,
        connection_wire: bytearray | None,
    ) -> None:
        """Read trailing headers after final chunk."""
        while True:
            line = await reader.readline()
            if not line:
                break
            self._extend_wire_buffers(line, wire, connection_wire)
            if line in (b"\r\n", b"\n", b""):
                break

    async def _read_chunked_body(
        self,
        reader: asyncio.StreamReader,
        wire: bytearray,
        connection_wire: bytearray | None = None,
    ) -> bytes:
        """Read a chunked transfer-encoded request body.

        * `wire` accumulates the *raw* chunk framing:
          - "<size>\\r\\n"
          - chunk bytes
          - "\\r\\n"
          - trailing headers
          - final blank line

        * return value is the decoded body (concatenation of chunk bytes).
        """
        body = bytearray()

        while True:
            line = await reader.readline()
            if not line:
                break
            self._extend_wire_buffers(line, wire, connection_wire)
            header = line.decode("ascii", errors="replace").strip()

            size = self._parse_chunk_size(header)
            if size is None:
                break

            if size == 0:
                await self._read_chunk_trailers(reader, wire, connection_wire)
                break

            # Read chunk data
            chunk = await reader.readexactly(size)
            body.extend(chunk)
            self._extend_wire_buffers(chunk, wire, connection_wire)

            # Read CRLF after chunk
            crlf = await reader.readexactly(2)
            self._extend_wire_buffers(crlf, wire, connection_wire)

        return bytes(body)

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
        response: StubResponse,
        body: bytes,
        should_close: bool,
    ) -> dict[str, str]:
        """Build complete response headers dict."""
        headers = dict(response.headers) if response.headers else {}
        header_names = {k.lower() for k in headers}

        if "content-length" not in header_names:
            headers["Content-Length"] = str(len(body))

        if "connection" not in header_names and should_close:
            headers["Connection"] = "close"

        return headers

    def _write_and_track(
        self,
        writer: asyncio.StreamWriter,
        data: bytes,
        connection_wire_sent: bytearray | None,
    ) -> None:
        """Write data to client and track in connection bytes."""
        writer.write(data)
        if connection_wire_sent is not None:
            connection_wire_sent.extend(data)

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        response: StubResponse,
        request: HTTPRequest,
        connection_wire_sent: bytearray | None = None,
    ) -> bool:
        try:
            reason = HTTPStatus(response.status).phrase
        except ValueError:
            reason = "UNKNOWN"

        status_line = f"HTTP/1.1 {response.status} {reason}\r\n"
        self._write_and_track(
            writer, status_line.encode("ascii"), connection_wire_sent
        )

        body = self._normalize_body(response.body)
        should_close = self._should_close_connection(request)
        headers = self._build_response_headers(response, body, should_close)

        for name, value in headers.items():
            header_line = f"{name}: {value}\r\n".encode("ascii")
            self._write_and_track(writer, header_line, connection_wire_sent)

        self._write_and_track(writer, b"\r\n", connection_wire_sent)
        self._write_and_track(writer, body, connection_wire_sent)
        await writer.drain()

        return should_close
