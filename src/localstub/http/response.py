from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from email.message import Message

import httptools


@dataclass
class RecordedResponse:
    """Captured HTTP response for recording/display purposes."""

    status: int
    reason: str | None
    headers: Message | None
    body: str | None
    wire_raw_bytes: bytes


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


def _is_close_delimited(
    headers: list[tuple[bytes, bytes]],
) -> bool:
    """Check if a response is close-delimited.

    A response is close-delimited if it has no Content-Length
    and no Transfer-Encoding header.  EOF signals the end of
    the body.
    """
    for name, _ in headers:
        name_lower = name.lower()
        if name_lower == b"content-length":
            return False
        if name_lower == b"transfer-encoding":
            return False
    return True


class ResponseProtocol:
    """Callback protocol for httptools.HttpResponseParser.

    All callbacks are guarded to be no-ops after a message is
    complete.  This prevents pipelined responses from overwriting
    the first completed response when httptools processes multiple
    responses in a single buffer.
    """

    def __init__(self) -> None:
        self.result = ParsedResponse()
        self._parser: httptools.HttpResponseParser | None = None
        self.message_complete = False

    def set_parser(self, parser: httptools.HttpResponseParser) -> None:
        """Set the parser reference for accessing metadata."""
        self._parser = parser

    def on_message_begin(self) -> None:
        if not self.message_complete:
            self.result = ParsedResponse()

    def on_status(self, status: bytes) -> None:
        if not self.message_complete:
            if self.result.status_text is None:
                self.result.status_text = status
            else:
                self.result.status_text += status

    def on_header(self, name: bytes, value: bytes) -> None:
        if not self.message_complete:
            self.result.headers.append((name, value))

    def on_headers_complete(self) -> None:
        if not self.message_complete and self._parser is not None:
            self.result.status_code = self._parser.get_status_code()
            self.result.http_version = self._parser.get_http_version()

    def on_body(self, body: bytes) -> None:
        if not self.message_complete:
            self.result.body_parts.append(body)

    def on_message_complete(self) -> None:
        if not self.message_complete:
            self.result.is_complete = True
            self.message_complete = True

    def on_chunk_header(self) -> None:
        pass

    def on_chunk_complete(self) -> None:
        pass


class AsyncResponseParser:
    """Async wrapper for httptools.HttpResponseParser with wire
    tracking.

    Combines asyncio stream reading with httptools parsing while
    preserving exact bytes received on the wire.  Handles
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
        request_method: str | None = None,
    ) -> tuple[ParsedResponse | None, bytes]:
        """Parse a complete HTTP response from the stream.

        Args:
            reader: The asyncio stream to read from.
            request_method: The HTTP method of the request that
                generated this response.  Required for HEAD
                requests, which have no body despite
                Content-Length headers.

        Returns:
            Tuple of (parsed_response, wire_bytes).
            Returns (None, wire_bytes) on parse error.
            For close-delimited bodies, EOF signals completion.
        """
        is_head = (
            request_method is not None and request_method.upper() == "HEAD"
        )

        while not self._protocol.result.is_complete:
            try:
                data = await reader.read(self._max_read)
            except Exception:
                break

            if not data:
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

            if is_head and self._protocol.result.http_version is not None:
                self._protocol.result.is_complete = True
                break

        if not self._protocol.result.is_complete:
            if self._protocol.result.http_version is not None:
                if _is_close_delimited(self._protocol.result.headers):
                    self._protocol.result.is_complete = True
                else:
                    return None, bytes(self._wire)
            else:
                return None, bytes(self._wire)

        return self._protocol.result, bytes(self._wire)

    @property
    def wire_bytes(self) -> bytes:
        """Return the accumulated wire bytes."""
        return bytes(self._wire)


class AsyncMultiResponseParser:
    """Parser that handles multiple HTTP responses from a stream.

    Designed for scenarios where a server sends one or more 1xx
    informational responses before the final response, and all
    responses may arrive in a single buffer read.

    Feeds data byte-by-byte to precisely track where each response
    ends, preserving exact wire bytes for each response.
    """

    def __init__(self, max_read: int = 8192) -> None:
        self._max_read = max_read
        self._buffer = bytearray()
        self._consumed = 0

    async def next_response(
        self,
        reader: asyncio.StreamReader,
        request_method: str | None = None,
    ) -> tuple[ParsedResponse | None, bytes]:
        """Parse and return the next complete HTTP response.

        Args:
            reader: The asyncio stream to read from.
            request_method: The HTTP method of the request.

        Returns:
            Tuple of (parsed_response, wire_bytes).
            Returns (None, wire_bytes) on parse error or EOF.
        """
        protocol = ResponseProtocol()
        parser = httptools.HttpResponseParser(protocol)
        protocol.set_parser(parser)

        is_head = (
            request_method is not None and request_method.upper() == "HEAD"
        )
        wire_start = self._consumed

        while True:
            while self._consumed < len(self._buffer):
                byte = bytes([self._buffer[self._consumed]])
                self._consumed += 1

                try:
                    parser.feed_data(byte)
                except httptools.HttpParserError:
                    wire = bytes(self._buffer[wire_start : self._consumed])
                    return None, wire

                if protocol.message_complete:
                    wire = bytes(self._buffer[wire_start : self._consumed])
                    return protocol.result, wire

                if is_head and protocol.result.http_version is not None:
                    protocol.result.is_complete = True
                    wire = bytes(self._buffer[wire_start : self._consumed])
                    return protocol.result, wire

            try:
                data = await reader.read(self._max_read)
            except Exception:
                break

            if not data:
                break

            self._buffer.extend(data)

        if protocol.result.http_version is not None:
            if _is_close_delimited(protocol.result.headers):
                protocol.result.is_complete = True
                wire = bytes(self._buffer[wire_start : self._consumed])
                return protocol.result, wire

        wire = bytes(self._buffer[wire_start : self._consumed])
        return None, wire
