import asyncio
import json
from dataclasses import dataclass, field
from email.message import Message
from typing import Any, Protocol

import httptools

from localstub.http.utils import headers_to_message


class Writer(Protocol):
    """Minimal StreamWriter interface for extracting client info."""

    def get_extra_info(self, name: str, default: Any | None = None) -> Any: ...


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


@dataclass
class ParsedRequest:
    """Intermediate representation of a parsed HTTP request."""

    method: str | None = None
    url: bytes | None = None
    http_version: str | None = None
    headers: list[tuple[bytes, bytes]] = field(default_factory=list)
    body_parts: list[bytes] = field(default_factory=list)
    is_complete: bool = False
    headers_complete: bool = False

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
            self.result.headers_complete = True

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

    async def parse_headers(
        self,
        reader: asyncio.StreamReader,
        connection_wire: bytearray | None = None,
    ) -> tuple[ParsedRequest | None, bytes, bytearray]:
        """Parse request headers only, stopping before body.

        Args:
            reader: The asyncio stream to read from.
            connection_wire: Optional buffer to accumulate connection-level
                bytes (for tracking across multiple requests).

        Returns:
            Tuple of (parsed_request, header_wire_bytes, remaining_buffer).
            The remaining_buffer contains bytes read but not yet processed,
            which should be passed to continue_parse_body().
            Returns (None, wire_bytes, empty_buffer) on parse error or EOF.
        """
        buffer = bytearray()

        while not self._protocol.result.headers_complete:
            if not buffer:
                try:
                    data = await reader.read(self._max_read)
                except Exception:
                    break

                if not data:
                    break

                buffer.extend(data)

            byte = bytes([buffer.pop(0)])
            self._wire.extend(byte)
            if connection_wire is not None:
                connection_wire.extend(byte)

            try:
                self._parser.feed_data(byte)
            except httptools.HttpParserError:
                if buffer:
                    reader.feed_data(bytes(buffer))
                return None, bytes(self._wire), bytearray()

        if not self._protocol.result.headers_complete:
            if buffer:
                reader.feed_data(bytes(buffer))
            return None, bytes(self._wire), bytearray()

        return self._protocol.result, bytes(self._wire), buffer

    async def continue_parse_body(
        self,
        reader: asyncio.StreamReader,
        remaining_buffer: bytearray,
        connection_wire: bytearray | None = None,
    ) -> tuple[ParsedRequest | None, bytes]:
        """Continue parsing the request body after headers.

        Call this after parse_headers() and after sending any interim
        response (like 100 Continue).

        Args:
            reader: The asyncio stream to continue reading from.
            remaining_buffer: Buffer returned from parse_headers().
            connection_wire: Optional buffer for connection-level tracking.

        Returns:
            Tuple of (parsed_request, complete_wire_bytes).
            Returns (None, wire_bytes) on parse error or EOF.
        """
        buffer = remaining_buffer

        while not self._protocol.result.is_complete:
            if not buffer:
                try:
                    data = await reader.read(self._max_read)
                except Exception:
                    break

                if not data:
                    break

                buffer.extend(data)

            byte = bytes([buffer.pop(0)])
            self._wire.extend(byte)
            if connection_wire is not None:
                connection_wire.extend(byte)

            try:
                self._parser.feed_data(byte)
            except httptools.HttpParserError:
                if buffer:
                    reader.feed_data(bytes(buffer))
                return None, bytes(self._wire)

        if buffer:
            reader.feed_data(bytes(buffer))

        if not self._protocol.result.is_complete:
            return None, bytes(self._wire)

        return self._protocol.result, bytes(self._wire)


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
