"""HTTP parsing module using httptools.

This module provides a clean separation between HTTP parsing logic and network
IO by using the httptools library (Python bindings to llhttp).

The architecture is:
- Protocol classes (RequestProtocol, ResponseProtocol): Pure parsing callbacks
- Async parser classes: Handle asyncio stream integration with wire tracking
- Utility functions: Convert parsed data to existing API types
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from email.message import Message
from typing import TYPE_CHECKING, Any, Protocol

import httptools

if TYPE_CHECKING:
    pass


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


class Writer(Protocol):
    """Minimal StreamWriter interface for extracting client info."""

    def get_extra_info(self, name: str, default: Any | None = None) -> Any: ...


@dataclass
class ParsedRequest:
    """Intermediate representation of a parsed HTTP request."""

    method: str | None = None
    url: bytes | None = None
    http_version: str | None = None
    headers: list[tuple[bytes, bytes]] = field(default_factory=list)
    body_parts: list[bytes] = field(default_factory=list)
    is_complete: bool = False

    @property
    def body(self) -> bytes:
        """Return the complete body as bytes."""
        return b"".join(self.body_parts)


@dataclass
class ParsedResponse:
    """Intermediate representation of a parsed HTTP response."""

    status_code: int | None = None
    status_text: bytes | None = None
    http_version: str | None = None
    headers: list[tuple[bytes, bytes]] = field(default_factory=list)
    body_parts: list[bytes] = field(default_factory=list)
    is_complete: bool = False

    @property
    def body(self) -> bytes:
        """Return the complete body as bytes."""
        return b"".join(self.body_parts)


class RequestProtocol:
    """Callback protocol for httptools.HttpRequestParser.

    All callbacks are guarded to be no-ops after a message is complete.
    This prevents pipelined requests from overwriting the first completed
    request when httptools processes multiple requests in a single buffer.
    """

    def __init__(self) -> None:
        self.result = ParsedRequest()
        self._parser: httptools.HttpRequestParser | None = None

    def set_parser(self, parser: httptools.HttpRequestParser) -> None:
        """Set the parser reference for accessing parsed metadata."""
        self._parser = parser

    def on_message_begin(self) -> None:
        """Called when a new message begins.

        Only resets if no message has been completed yet. This preserves
        the first request when pipelined requests arrive in the same buffer.
        """
        if not self.result.is_complete:
            self.result = ParsedRequest()

    def on_url(self, url: bytes) -> None:
        """Called when the URL is parsed.

        Note: May be called multiple times with URL chunks when data
        arrives incrementally. We accumulate all chunks.
        """
        if not self.result.is_complete:
            if self.result.url is None:
                self.result.url = url
            else:
                self.result.url += url

    def on_header(self, name: bytes, value: bytes) -> None:
        """Called for each header."""
        if not self.result.is_complete:
            self.result.headers.append((name, value))

    def on_headers_complete(self) -> None:
        """Called when all headers have been parsed."""
        if not self.result.is_complete and self._parser is not None:
            self.result.method = self._parser.get_method().decode(
                "ascii", errors="replace"
            )
            self.result.http_version = self._parser.get_http_version()

    def on_body(self, body: bytes) -> None:
        """Called for each chunk of body data."""
        if not self.result.is_complete:
            self.result.body_parts.append(body)

    def on_message_complete(self) -> None:
        """Called when the message is fully parsed."""
        if not self.result.is_complete:
            self.result.is_complete = True

    def on_chunk_header(self) -> None:
        """Called at the start of a chunk (for chunked encoding)."""
        pass

    def on_chunk_complete(self) -> None:
        """Called at the end of a chunk (for chunked encoding)."""
        pass


class ResponseProtocol:
    """Callback protocol for httptools.HttpResponseParser.

    All callbacks are guarded to be no-ops after a message is complete.
    This prevents pipelined responses from overwriting the first completed
    response when httptools processes multiple responses in a single buffer.
    """

    def __init__(self) -> None:
        self.result = ParsedResponse()
        self._parser: httptools.HttpResponseParser | None = None

    def set_parser(self, parser: httptools.HttpResponseParser) -> None:
        """Set the parser reference for accessing parsed metadata."""
        self._parser = parser

    def on_message_begin(self) -> None:
        """Called when a new message begins.

        Only resets if no message has been completed yet. This preserves
        the first response when pipelined responses arrive in the same buffer.
        """
        if not self.result.is_complete:
            self.result = ParsedResponse()

    def on_status(self, status: bytes) -> None:
        """Called when the status text is parsed.

        Note: May be called multiple times with status chunks when data
        arrives incrementally. We accumulate all chunks.
        """
        if not self.result.is_complete:
            if self.result.status_text is None:
                self.result.status_text = status
            else:
                self.result.status_text += status

    def on_header(self, name: bytes, value: bytes) -> None:
        """Called for each header."""
        if not self.result.is_complete:
            self.result.headers.append((name, value))

    def on_headers_complete(self) -> None:
        """Called when all headers have been parsed."""
        if not self.result.is_complete and self._parser is not None:
            self.result.status_code = self._parser.get_status_code()
            self.result.http_version = self._parser.get_http_version()

    def on_body(self, body: bytes) -> None:
        """Called for each chunk of body data."""
        if not self.result.is_complete:
            self.result.body_parts.append(body)

    def on_message_complete(self) -> None:
        """Called when the message is fully parsed."""
        if not self.result.is_complete:
            self.result.is_complete = True

    def on_chunk_header(self) -> None:
        """Called at the start of a chunk (for chunked encoding)."""
        pass

    def on_chunk_complete(self) -> None:
        """Called at the end of a chunk (for chunked encoding)."""
        pass


def headers_to_message(headers: list[tuple[bytes, bytes]]) -> Message:
    """Convert parsed headers to email.message.Message for API compatibility.

    Args:
        headers: List of (name, value) byte tuples from the parser.

    Returns:
        Message object with headers accessible via dict-like interface.
    """
    msg = Message()
    for name, value in headers:
        msg[name.decode("iso-8859-1")] = value.decode("iso-8859-1")
    return msg


class AsyncRequestParser:
    """Async wrapper for httptools.HttpRequestParser with wire tracking.

    This class combines asyncio stream reading with httptools parsing while
    preserving the exact bytes received on the wire. It correctly handles
    pipelined requests by feeding data byte-by-byte to detect message
    boundaries and pushing leftover bytes back to the reader.
    """

    def __init__(self, max_read: int = 8192) -> None:
        self._protocol = RequestProtocol()
        self._parser = httptools.HttpRequestParser(self._protocol)
        self._protocol.set_parser(self._parser)
        self._wire = bytearray()
        self._max_read = max_read

    async def parse(
        self,
        reader: asyncio.StreamReader,
        connection_wire: bytearray | None = None,
    ) -> tuple[ParsedRequest | None, bytes]:
        """Parse a complete HTTP request from the stream.

        Args:
            reader: The asyncio stream to read from.
            connection_wire: Optional buffer to accumulate connection-level
                bytes (for tracking across multiple requests).

        Returns:
            Tuple of (parsed_request, wire_bytes).
            Returns (None, wire_bytes) on parse error or EOF before complete.

        Note:
            This method feeds data byte-by-byte to detect message boundaries
            precisely. Any bytes belonging to a subsequent pipelined request
            are pushed back into the reader for the next parse() call.
        """
        buffer = bytearray()

        while not self._protocol.result.is_complete:
            # If buffer is empty, read more data from the stream
            if not buffer:
                try:
                    data = await reader.read(self._max_read)
                except Exception:
                    break

                if not data:
                    # EOF before complete message
                    break

                buffer.extend(data)

            # Feed one byte at a time to detect message boundary precisely
            byte = bytes([buffer.pop(0)])
            self._wire.extend(byte)
            if connection_wire is not None:
                connection_wire.extend(byte)

            try:
                self._parser.feed_data(byte)
            except httptools.HttpParserError:
                # Malformed request - push remaining buffer back for recovery
                if buffer:
                    reader.feed_data(bytes(buffer))
                return None, bytes(self._wire)

        # Push any leftover bytes back to the reader for the next request
        if buffer:
            reader.feed_data(bytes(buffer))

        if not self._protocol.result.is_complete:
            return None, bytes(self._wire)

        return self._protocol.result, bytes(self._wire)

    @property
    def wire_bytes(self) -> bytes:
        """Return the accumulated wire bytes."""
        return bytes(self._wire)


class AsyncResponseParser:
    """Async wrapper for httptools.HttpResponseParser with wire tracking.

    This class combines asyncio stream reading with httptools parsing while
    preserving the exact bytes received on the wire. It handles both
    Content-Length, chunked, and close-delimited response bodies.
    """

    def __init__(self, max_read: int = 8192) -> None:
        self._protocol = ResponseProtocol()
        self._parser = httptools.HttpResponseParser(self._protocol)
        self._protocol.set_parser(self._parser)
        self._wire = bytearray()
        self._max_read = max_read

    async def parse(
        self,
        reader: asyncio.StreamReader,
    ) -> tuple[ParsedResponse | None, bytes]:
        """Parse a complete HTTP response from the stream.

        Args:
            reader: The asyncio stream to read from.

        Returns:
            Tuple of (parsed_response, wire_bytes).
            Returns (None, wire_bytes) on parse error.
            For close-delimited bodies, EOF signals completion.
        """
        while not self._protocol.result.is_complete:
            try:
                data = await reader.read(self._max_read)
            except Exception:
                break

            if not data:
                # EOF - for close-delimited responses, this signals completion
                # Feed empty data to finalize the parser state
                try:
                    self._parser.feed_data(b"")
                except httptools.HttpParserError:
                    pass
                break

            self._wire.extend(data)

            try:
                self._parser.feed_data(data)
            except httptools.HttpParserError:
                return None, bytes(self._wire)

        # For close-delimited bodies, the message may be complete after EOF
        # even if on_message_complete wasn't called
        if not self._protocol.result.is_complete:
            # If we have headers and got EOF, treat as close-delimited
            if self._protocol.result.http_version is not None:
                self._protocol.result.is_complete = True
            else:
                return None, bytes(self._wire)

        return self._protocol.result, bytes(self._wire)

    @property
    def wire_bytes(self) -> bytes:
        """Return the accumulated wire bytes."""
        return bytes(self._wire)


class HTTPRequestReader:
    """Public interface for reading HTTP requests from streams.

    This provides a clean way to parse HTTP requests without needing
    an AsyncHTTPTestServer instance.
    """

    def __init__(self, max_read: int = 8192) -> None:
        self._max_read = max_read

    async def read_request(
        self,
        reader: asyncio.StreamReader,
        writer: Writer | None = None,
        connection_wire: bytearray | None = None,
    ) -> HTTPRequest | None:
        """Parse a complete HTTP request from the stream.

        Args:
            reader: The asyncio stream to read from
            writer: Optional writer to extract client info from
            connection_wire: Optional buffer for connection-level tracking

        Returns:
            HTTPRequest or None on EOF/parse error
        """
        parser = AsyncRequestParser(max_read=self._max_read)
        parsed, wire_bytes = await parser.parse(reader, connection_wire)

        if parsed is None:
            return None

        headers = headers_to_message(parsed.headers)
        body_text = (
            parsed.body.decode("utf-8", errors="replace")
            if parsed.body
            else None
        )
        client = self._extract_client_info(writer) if writer else None

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

    def _extract_client_info(self, writer: Writer) -> tuple[str, int] | None:
        """Extract client (host, port) from the writer's peername."""
        peer = writer.get_extra_info("peername")
        if isinstance(peer, tuple) and len(peer) >= 2:
            return (peer[0], peer[1])
        return None
