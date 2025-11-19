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
class RequestRecorder:
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


Handler = Callable[[RequestRecorder], Awaitable[StubResponse] | StubResponse]


# ---------------------------------------------------------------------------
# Async HTTP server
# ---------------------------------------------------------------------------


class AsyncHTTPTestServer:
    """Small asyncio HTTP server used for testing SDK clients.

    Features:
      * exposes .url (e.g. "http://127.0.0.1:12345/")
      * records last_request (RequestRecorder) and a list of all requests
      * `wire_raw_bytes` contains the *exact* bytes received, including
        chunked framing and trailers.
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

        self.last_request: RequestRecorder | None = None
        self.requests: list[RequestRecorder] = []
        self._request_queue: asyncio.Queue[RequestRecorder] = asyncio.Queue()

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

    async def next_request(
        self, timeout: float | None = None
    ) -> RequestRecorder:
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

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                request = await self._read_request(reader, writer)
                if request is None:
                    break

                self.last_request = request
                self.requests.append(request)
                await self._request_queue.put(request)

                if self._handler is None:
                    response = self._default_response
                else:
                    result = self._handler(request)
                    if inspect.isawaitable(result):
                        response = await result  # type: ignore[assignment]
                    else:
                        response = result

                should_close = await self._write_response(
                    writer, response, request
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
    ) -> RequestRecorder | None:
        wire = bytearray()
        header_lines: list[bytes] = []

        # --- Request line ---
        req_line_parts = await self._read_request_line(reader, wire)
        if req_line_parts is None:
            return None
        method, path, version = req_line_parts

        # --- Headers ---
        headers = await self._read_headers(reader, wire, header_lines)
        if headers is None:
            return None

        # --- Body ---
        body_text = await self._read_body(reader, wire, headers)

        # --- Client info ---
        client = self._extract_client_info(writer)

        return RequestRecorder(
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
    ) -> tuple[str, str, str] | None:
        """Read and parse the HTTP request line."""
        line = await reader.readline()
        if not line:
            return None
        wire.extend(line)
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
    ) -> Message | None:
        """Read HTTP headers until blank line."""
        while True:
            line = await reader.readline()
            if not line:
                # EOF while reading headers
                return None
            wire.extend(line)
            if line in (b"\r\n", b"\n"):
                break
            header_lines.append(line)
        return _parse_headers(header_lines)

    async def _read_body(
        self,
        reader: asyncio.StreamReader,
        wire: bytearray,
        headers: Message,
    ) -> str | None:
        """Read request body based on Transfer-Encoding or Content-Length."""
        body_bytes: bytes | None = None
        transfer_encoding = headers.get("Transfer-Encoding")
        if transfer_encoding and "chunked" in transfer_encoding.lower():
            body_bytes = await self._read_chunked_body(reader, wire)
        else:
            body_bytes = await self._read_content_length_body(
                reader, wire, headers
            )

        if body_bytes is None:
            return None
        return body_bytes.decode("utf-8", errors="replace")

    async def _read_content_length_body(
        self,
        reader: asyncio.StreamReader,
        wire: bytearray,
        headers: Message,
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

    async def _read_chunked_body(
        self,
        reader: asyncio.StreamReader,
        wire: bytearray,
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
            # Chunk-size line
            line = await reader.readline()
            if not line:
                break
            wire.extend(line)
            header = line.decode("ascii", errors="replace").strip()

            # Handle aws-chunked style: "1a;chunk-signature=..."
            if ";" in header:
                size_str = header.split(";", 1)[0]
            else:
                size_str = header

            try:
                size = int(size_str, 16)
            except ValueError:
                LOG.debug("Invalid chunk size header: %r", header)
                break

            if size == 0:
                # Trailing headers (if any), ending with blank line
                while True:
                    line = await reader.readline()
                    if not line:
                        break
                    wire.extend(line)
                    if line in (b"\r\n", b"\n", b""):
                        break
                break

            # Chunk data
            chunk = await reader.readexactly(size)
            body.extend(chunk)
            wire.extend(chunk)

            # CRLF after each chunk
            crlf = await reader.readexactly(2)
            wire.extend(crlf)

        return bytes(body)

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        response: StubResponse,
        request: RequestRecorder,
    ) -> bool:
        try:
            reason = HTTPStatus(response.status).phrase
        except ValueError:
            reason = "UNKNOWN"

        status_line = f"HTTP/1.1 {response.status} {reason}\r\n"
        writer.write(status_line.encode("ascii"))

        # Normalize body to bytes
        body_obj = response.body
        if body_obj is None:
            body = b""
        elif isinstance(body_obj, bytes):
            body = body_obj
        else:
            body = str(body_obj).encode("utf-8")

        headers = (
            dict(response.headers) if response.headers is not None else {}
        )
        header_names = {k.lower() for k in headers}
        if "content-length" not in header_names:
            headers["Content-Length"] = str(len(body))

        # Check if client requested connection close
        client_connection = (
            request.headers.get("Connection", "").lower()
            if request.headers
            else ""
        )
        should_close = "close" in client_connection

        # Set Connection header if not already set by user
        if "connection" not in header_names:
            if should_close:
                headers["Connection"] = "close"

        for name, value in headers.items():
            writer.write(f"{name}: {value}\r\n".encode("ascii"))
        writer.write(b"\r\n")
        writer.write(body)
        await writer.drain()

        return should_close
