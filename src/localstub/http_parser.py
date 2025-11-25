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
from dataclasses import dataclass, field
from email.message import Message
from typing import TYPE_CHECKING

import httptools

if TYPE_CHECKING:
    pass


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
        """Called when the URL is parsed."""
        if not self.result.is_complete:
            self.result.url = url

    def on_header(self, name: bytes, value: bytes) -> None:
        """Called for each header."""
        if not self.result.is_complete:
            self.result.headers.append((name, value))

    def on_headers_complete(self) -> None:
        """Called when all headers have been parsed."""
        if not self.result.is_complete and self._parser is not None:
            self.result.method = self._parser.get_method().decode("ascii")
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
        """Called when the status text is parsed."""
        if not self.result.is_complete:
            self.result.status_text = status

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
    preserving the exact bytes received on the wire.
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
        """
        while not self._protocol.result.is_complete:
            try:
                data = await reader.read(self._max_read)
            except Exception:
                break

            if not data:
                # EOF before complete message
                break

            # Track wire bytes BEFORE parsing to preserve exact format
            self._wire.extend(data)
            if connection_wire is not None:
                connection_wire.extend(data)

            try:
                self._parser.feed_data(data)
            except httptools.HttpParserError:
                # Malformed request
                return None, bytes(self._wire)

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
