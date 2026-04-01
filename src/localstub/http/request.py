from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import httptools

from localstub.http.headers import Headers
from localstub.http.uri import ParsedURI, parse_absolute_uri
from localstub.http.utils import headers_to_headers

HEADER_TERMINATOR = b"\r\n\r\n"
CRLF = b"\r\n"


class Writer(Protocol):
    """Minimal StreamWriter interface for extracting client info."""

    def get_extra_info(self, name: str, default: Any | None = None) -> Any: ...


@dataclass(frozen=True)
class HTTPRequest:
    """Snapshot of a single HTTP request.

    `body` is decoded as UTF-8 (like the original code).
    `body_bytes` is the parsed body bytes (may differ from wire framing).
    `wire_raw_bytes` is *exactly* what came off the wire, including:
      - request line
      - headers
      - the blank line
      - body bytes (including chunk framing for chunked requests)
    """

    method: str
    path: str
    http_version: str
    headers: Headers = field(default_factory=Headers.empty)
    body: str = ""
    body_bytes: bytes = b""
    wire_raw_bytes: bytes = b""
    client: tuple[str, int] | None = None

    @classmethod
    def from_parsed(
        cls,
        parsed: ParsedRequest,
        wire_raw_bytes: bytes,
        *,
        writer: Writer | None = None,
    ) -> HTTPRequest:
        headers = headers_to_headers(parsed.headers)
        body_bytes = parsed.body
        body_text = body_bytes.decode("utf-8", errors="replace")
        path = (
            parsed.url.decode("ascii", errors="replace")
            if parsed.url is not None
            else ""
        )
        client: tuple[str, int] | None = None
        if writer is not None:
            peer = writer.get_extra_info("peername")
            if isinstance(peer, tuple) and len(peer) >= 2:
                client = (peer[0], peer[1])

        return cls(
            method=parsed.method or "",
            path=path,
            http_version=parsed.http_version or "",
            headers=headers,
            body=body_text,
            body_bytes=body_bytes,
            wire_raw_bytes=wire_raw_bytes,
            client=client,
        )

    def with_path(self, path: str) -> HTTPRequest:
        return replace(self, path=path)

    def with_method(self, method: str) -> HTTPRequest:
        return replace(self, method=method)

    def with_headers(self, headers: Headers) -> HTTPRequest:
        return replace(self, headers=headers)

    @property
    def wire_body_bytes(self) -> bytes:
        """Return request body bytes as they appeared on the wire.

        For chunked uploads this includes the original chunk
        framing and any trailer bytes.
        """
        if self.body_bytes:
            body_fallback = self.body_bytes
        elif self.body:
            body_fallback = self.body.encode()
        else:
            body_fallback = b""
        header_end = self.wire_raw_bytes.find(b"\r\n\r\n")
        if header_end == -1:
            return body_fallback

        return self.wire_raw_bytes[header_end + 4 :]

    @property
    def json_body(self) -> Any:
        if self.body == "":
            return None
        return json.loads(self.body)

    @property
    def is_proxy_request(self) -> bool:
        """Return True if this is a forward proxy request (absolute-form URI).

        Forward proxy requests have the full URL in the request line,
        e.g., GET http://example.com/path HTTP/1.1
        """
        return self.path.startswith("http://") or self.path.startswith(
            "https://"
        )

    @property
    def target_uri(self) -> ParsedURI | None:
        """Parse and return URI components if absolute-form, else None.

        Returns:
            ParsedURI with scheme, host, port, path for absolute-form URIs,
            or None for origin-form URIs (e.g., "/path").
        """
        return parse_absolute_uri(self.path)

    @property
    def effective_path(self) -> str:
        """Return the path portion for routing.

        For absolute-form URIs (e.g., "http://example.com/foo?bar=1"),
        returns just the path and query string ("/foo?bar=1").

        For origin-form URIs (e.g., "/foo"), returns the path as-is.

        This is useful for route matching where you want
        add_route("GET", "/foo", handler) to match both origin-form
        and absolute-form requests.
        """
        if not self.path:
            return "/"
        uri = self.target_uri
        if uri is not None:
            return uri.path
        return self.path


@dataclass(frozen=True)
class HTTPRequestHeaders:
    """Partial request available after headers are parsed, before body."""

    method: str | None
    path: str | None
    http_version: str | None
    headers: Headers
    wire_raw_bytes: bytes


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


class _ChunkScanError(Exception):
    def __init__(self, offset: int) -> None:
        self.offset = offset


def _build_request_parser() -> tuple[
    RequestProtocol, httptools.HttpRequestParser
]:
    protocol = RequestProtocol()
    parser = httptools.HttpRequestParser(protocol)
    protocol.set_parser(parser)
    return protocol, parser


def _is_hex_digit(value: int) -> bool:
    return (
        ord("0") <= value <= ord("9")
        or ord("A") <= value <= ord("F")
        or ord("a") <= value <= ord("f")
    )


def _content_length(
    headers: list[tuple[bytes, bytes]],
) -> int | None:
    for name, value in reversed(headers):
        if name.lower() != b"content-length":
            continue
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _is_chunked_transfer(
    headers: list[tuple[bytes, bytes]],
) -> bool:
    for name, value in headers:
        if name.lower() != b"transfer-encoding":
            continue
        for token in value.split(b","):
            if token.strip().lower() == b"chunked":
                return True
    return False


def _parse_chunk_size(
    buffer: bytearray,
    start: int,
    end: int,
) -> int:
    if start == end:
        raise _ChunkScanError(start + 1)
    return int(bytes(buffer[start:end]), 16)


def _scan_chunk_size_line(
    buffer: bytearray,
    start: int,
) -> tuple[int, int] | None:
    size_end = start

    while True:
        if size_end >= len(buffer):
            return None

        current = buffer[size_end]
        if _is_hex_digit(current):
            size_end += 1
            continue

        if current == ord(";"):
            line_end = buffer.find(CRLF, size_end)
            if line_end == -1:
                return None
            return _parse_chunk_size(buffer, start, size_end), line_end + 2

        if current == ord("\r"):
            if size_end + 1 >= len(buffer):
                return None
            if buffer[size_end + 1] != ord("\n"):
                raise _ChunkScanError(size_end + 1)
            return _parse_chunk_size(buffer, start, size_end), size_end + 2

        raise _ChunkScanError(size_end + 1)


def _scan_chunked_body_end(buffer: bytearray) -> int | None:
    index = 0

    while True:
        chunk_size_line = _scan_chunk_size_line(buffer, index)
        if chunk_size_line is None:
            return None

        chunk_size, chunk_data_start = chunk_size_line

        if chunk_size == 0:
            if buffer[chunk_data_start : chunk_data_start + 2] == CRLF:
                return chunk_data_start + 2
            trailer_end = buffer.find(HEADER_TERMINATOR, chunk_data_start)
            if trailer_end == -1:
                return None
            return trailer_end + len(HEADER_TERMINATOR)

        chunk_data_end = chunk_data_start + chunk_size
        if chunk_data_end + 2 > len(buffer):
            return None
        if buffer[chunk_data_end : chunk_data_end + 2] != CRLF:
            mismatch = chunk_data_end
            while mismatch < len(buffer) and mismatch < chunk_data_end + 2:
                expected = (
                    ord("\r") if mismatch == chunk_data_end else ord("\n")
                )
                if buffer[mismatch] != expected:
                    raise _ChunkScanError(mismatch + 1)
                mismatch += 1
            return None

        index = chunk_data_end + 2


class AsyncRequestParser:
    """Async wrapper for httptools.HttpRequestParser with wire tracking.

    This class combines asyncio stream reading with httptools parsing while
    preserving the exact bytes received on the wire. It correctly handles
    pipelined requests by feeding data byte-by-byte to detect message
    boundaries and pushing leftover bytes back to the reader.
    """

    def __init__(self, max_read: int = 8192) -> None:
        self._protocol, self._parser = _build_request_parser()
        self._wire = bytearray()
        self._connection_wire_offset = 0
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
            This method parses headers and bodies in buffered segments while
            preserving exact wire bytes. Any bytes belonging to a subsequent
            pipelined request are pushed back into the reader for the next
            parse() call.
        """
        parsed, wire_bytes, remaining = await self.parse_headers(
            reader,
            connection_wire,
        )
        if parsed is None:
            return None, wire_bytes

        if parsed.is_complete:
            self._push_back(reader, remaining)
            return parsed, wire_bytes

        return await self.continue_parse_body(
            reader,
            remaining,
            connection_wire,
        )

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
        fed = 0

        while not self._protocol.result.headers_complete:
            header_end = buffer.find(HEADER_TERMINATOR)
            if header_end == -1:
                has_more = await self._read_more(reader, buffer)
                if not has_more:
                    break
                header_end = buffer.find(HEADER_TERMINATOR)

            feed_end = (
                header_end + len(HEADER_TERMINATOR)
                if header_end != -1
                else len(buffer)
            )
            if feed_end == fed:
                break

            segment = bytes(buffer[fed:feed_end])
            try:
                self._parser.feed_data(segment)
            except httptools.HttpParserError:
                error_offset = self._precise_error_offset(
                    bytes(buffer[:feed_end])
                )
                self._wire.extend(buffer[:error_offset])
                self._sync_connection_wire(connection_wire)
                self._push_back(reader, buffer[error_offset:])
                return None, bytes(self._wire), bytearray()

            fed = feed_end

        if not self._protocol.result.headers_complete:
            self._wire.extend(buffer)
            self._sync_connection_wire(connection_wire)
            return None, bytes(self._wire), bytearray()

        header_end = buffer.find(HEADER_TERMINATOR) + len(HEADER_TERMINATOR)
        self._wire.extend(buffer[:header_end])
        self._sync_connection_wire(connection_wire)
        return (
            self._protocol.result,
            bytes(self._wire),
            buffer[header_end:],
        )

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

        if self._protocol.result.is_complete:
            self._push_back(reader, buffer)
            self._sync_connection_wire(connection_wire)
            return self._protocol.result, bytes(self._wire)

        if _is_chunked_transfer(self._protocol.result.headers):
            return await self._parse_chunked_body(
                reader,
                buffer,
                connection_wire,
            )

        content_length = _content_length(self._protocol.result.headers)
        if content_length is not None:
            return await self._parse_content_length_body(
                reader,
                buffer,
                content_length,
                connection_wire,
            )

        while not self._protocol.result.is_complete:
            if not buffer:
                has_more = await self._read_more(reader, buffer)
                if not has_more:
                    break

            segment = bytes(buffer)
            try:
                self._parser.feed_data(segment)
            except httptools.HttpParserError:
                return self._body_parse_error(
                    reader,
                    buffer,
                    len(buffer),
                    connection_wire,
                )

            self._wire.extend(segment)
            buffer.clear()

        if not self._protocol.result.is_complete:
            self._wire.extend(buffer)
            self._sync_connection_wire(connection_wire)
            return None, bytes(self._wire)

        self._push_back(reader, buffer)
        self._sync_connection_wire(connection_wire)
        return self._protocol.result, bytes(self._wire)

    async def _read_more(
        self,
        reader: asyncio.StreamReader,
        buffer: bytearray,
    ) -> bool:
        try:
            data = await reader.read(self._max_read)
        except Exception:
            return False

        if not data:
            return False

        buffer.extend(data)
        return True

    def _sync_connection_wire(
        self,
        connection_wire: bytearray | None,
    ) -> None:
        if connection_wire is None:
            return

        if self._connection_wire_offset >= len(self._wire):
            return

        connection_wire.extend(self._wire[self._connection_wire_offset :])
        self._connection_wire_offset = len(self._wire)

    def _push_back(
        self,
        reader: asyncio.StreamReader,
        buffer: bytes | bytearray,
    ) -> None:
        if buffer:
            reader.feed_data(bytes(buffer))

    def _precise_error_offset(self, data: bytes) -> int:
        protocol, parser = _build_request_parser()

        for index, value in enumerate(data, start=1):
            try:
                parser.feed_data(bytes((value,)))
            except httptools.HttpParserError:
                return index

            if protocol.result.is_complete:
                return index

        return len(data)

    def _body_parse_error(
        self,
        reader: asyncio.StreamReader,
        buffer: bytearray,
        consumed_guess: int,
        connection_wire: bytearray | None,
    ) -> tuple[ParsedRequest | None, bytes]:
        candidate = bytes(self._wire) + bytes(buffer[:consumed_guess])
        error_offset = self._precise_error_offset(candidate)
        body_offset = max(0, error_offset - len(self._wire))
        self._wire.extend(buffer[:body_offset])
        self._sync_connection_wire(connection_wire)
        self._push_back(reader, buffer[body_offset:])
        return None, bytes(self._wire)

    async def _parse_content_length_body(
        self,
        reader: asyncio.StreamReader,
        buffer: bytearray,
        content_length: int,
        connection_wire: bytearray | None,
    ) -> tuple[ParsedRequest | None, bytes]:
        while len(buffer) < content_length:
            has_more = await self._read_more(reader, buffer)
            if not has_more:
                self._wire.extend(buffer)
                self._sync_connection_wire(connection_wire)
                return None, bytes(self._wire)

        body = bytes(buffer[:content_length])
        try:
            self._parser.feed_data(body)
        except httptools.HttpParserError:
            return self._body_parse_error(
                reader,
                buffer,
                content_length,
                connection_wire,
            )

        self._wire.extend(body)
        self._push_back(reader, buffer[content_length:])
        self._sync_connection_wire(connection_wire)

        if not self._protocol.result.is_complete:
            return None, bytes(self._wire)

        return self._protocol.result, bytes(self._wire)

    async def _parse_chunked_body(
        self,
        reader: asyncio.StreamReader,
        buffer: bytearray,
        connection_wire: bytearray | None,
    ) -> tuple[ParsedRequest | None, bytes]:
        while True:
            try:
                chunked_end = _scan_chunked_body_end(buffer)
            except _ChunkScanError as exc:
                self._wire.extend(buffer[: exc.offset])
                self._sync_connection_wire(connection_wire)
                self._push_back(reader, buffer[exc.offset :])
                return None, bytes(self._wire)

            if chunked_end is not None:
                chunked_body = bytes(buffer[:chunked_end])
                try:
                    self._parser.feed_data(chunked_body)
                except httptools.HttpParserError:
                    return self._body_parse_error(
                        reader,
                        buffer,
                        chunked_end,
                        connection_wire,
                    )

                self._wire.extend(chunked_body)
                self._push_back(reader, buffer[chunked_end:])
                self._sync_connection_wire(connection_wire)

                if not self._protocol.result.is_complete:
                    return None, bytes(self._wire)

                return self._protocol.result, bytes(self._wire)

            has_more = await self._read_more(reader, buffer)
            if not has_more:
                self._wire.extend(buffer)
                self._sync_connection_wire(connection_wire)
                return None, bytes(self._wire)


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

        return HTTPRequest.from_parsed(parsed, wire_bytes, writer=writer)


def parsed_body_bytes_from_wire_raw_bytes(
    wire_raw_bytes: bytes,
) -> bytes | None:
    """Parse wire request bytes and return the decoded body bytes.

    This returns the body bytes as produced by the HTTP parser. For chunked
    uploads, this is the de-chunked body (it does not include chunk framing
    or trailer bytes).

    Returns None if the wire bytes cannot be parsed as a complete request.
    """
    if not wire_raw_bytes:
        return None

    protocol = RequestProtocol()
    parser = httptools.HttpRequestParser(protocol)
    protocol.set_parser(parser)
    try:
        parser.feed_data(wire_raw_bytes)
    except httptools.HttpParserError:
        return None

    if not protocol.result.is_complete:
        return None

    return protocol.result.body
