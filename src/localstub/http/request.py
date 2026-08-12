from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import httptools

from localstub.http import stream
from localstub.http.framing import (
    HEADER_TERMINATOR,
    ChunkScanError,
    content_length,
    is_chunked_transfer,
    scan_chunked_body,
)
from localstub.http.headers import HeaderItem, Headers
from localstub.http.uri import ParsedURI, parse_absolute_uri
from localstub.http.utils import headers_to_headers

LOG = logging.getLogger(__name__)


class Writer(Protocol):
    """Minimal StreamWriter interface for extracting client info."""

    def get_extra_info(self, name: str, default: Any | None = None) -> Any: ...


_FRAMING_HEADERS = frozenset({"content-length", "transfer-encoding"})


def _semantic_header_items(headers: Headers) -> tuple[HeaderItem, ...]:
    """Header items with framing headers removed, for value equality."""
    return tuple(
        (name, value)
        for name, value in headers.items()
        if name.lower() not in _FRAMING_HEADERS
    )


@dataclass(frozen=True, eq=False)
class HTTPRequest:
    """An HTTP request: what RFC 9110 says a request is, nothing else.

    ``target`` is the request-target from the request line: origin-form
    ("/v1/items?x=1") or absolute-form ("http://host/...").

    ``body`` distinguishes no content (``None``, no content headers at
    all) from empty content (``b""``, e.g. ``Content-Length: 0``).

    Equality is semantic: the framing headers ``Content-Length`` and
    ``Transfer-Encoding`` are excluded from comparison and hashing, so
    two requests compare equal even if one arrived chunked and one with
    Content-Length.  Framing evidence lives on ``RecordedHTTPRequest``.
    """

    method: str
    target: str
    headers: Headers = field(default_factory=Headers.empty)
    body: bytes | None = None

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HTTPRequest):
            return NotImplemented
        return (
            self.method == other.method
            and self.target == other.target
            and self.body == other.body
            and _semantic_header_items(self.headers)
            == _semantic_header_items(other.headers)
        )

    def __hash__(self) -> int:
        return hash((
            self.method,
            self.target,
            self.body,
            _semantic_header_items(self.headers),
        ))

    @property
    def text(self) -> str:
        """Return the body decoded as UTF-8 with replacement."""
        if self.body is None:
            return ""
        return self.body.decode("utf-8", errors="replace")

    @property
    def json_body(self) -> Any:
        if not self.body:
            return None
        return json.loads(self.body)

    @property
    def is_proxy_request(self) -> bool:
        """Return True if this is a forward proxy request (absolute-form URI).

        Forward proxy requests have the full URL in the request line,
        e.g., GET http://example.com/path HTTP/1.1
        """
        return self.target.startswith("http://") or self.target.startswith(
            "https://"
        )

    @property
    def target_uri(self) -> ParsedURI | None:
        """Parse and return URI components if absolute-form, else None.

        Returns:
            ParsedURI with scheme, host, port, path for absolute-form URIs,
            or None for origin-form URIs (e.g., "/path").
        """
        return parse_absolute_uri(self.target)

    @property
    def effective_path(self) -> str:
        """Return the path portion for routing.

        For absolute-form URIs (e.g., "http://example.com/foo?bar=1"),
        returns just the path and query string ("/foo?bar=1").

        For origin-form URIs (e.g., "/foo"), returns the target as-is.

        This is useful for route matching where you want
        add_route("GET", "/foo", handler) to match both origin-form
        and absolute-form requests.
        """
        if not self.target:
            return "/"
        uri = self.target_uri
        if uri is not None:
            return uri.path
        return self.target


def _parsed_body(parsed: ParsedRequest) -> bytes | None:
    """Map a parsed body to HTTPRequest.body semantics (absent → None)."""
    if parsed.body_parts:
        return parsed.body
    if is_chunked_transfer(parsed.headers):
        return b""
    if content_length(parsed.headers) is not None:
        return b""
    return None


@dataclass(frozen=True)
class RecordedHTTPRequest:
    """An HTTPRequest we witnessed on the wire, plus the evidence.

    ``request`` carries the current semantics (middleware rewrites
    included); ``as_received`` carries the semantics exactly as parsed
    off the wire.  ``wire_raw_bytes`` is *exactly* what came off the
    wire, including:
      - request line
      - headers
      - the blank line
      - body bytes (including chunk framing for chunked requests)

    Reads delegate to ``request``; writes (``with_method`` etc.) rebuild
    ``request`` while ``as_received`` and the wire evidence never change
    after construction.
    """

    request: HTTPRequest
    as_received: HTTPRequest
    wire_raw_bytes: bytes
    http_version: str
    client: tuple[str, int] | None = None

    @classmethod
    def from_parsed(
        cls,
        parsed: ParsedRequest,
        wire_raw_bytes: bytes,
        *,
        client: tuple[str, int] | None = None,
        writer: Writer | None = None,
    ) -> RecordedHTTPRequest:
        target = (
            parsed.url.decode("ascii", errors="replace")
            if parsed.url is not None
            else ""
        )
        if client is None and writer is not None:
            peer = writer.get_extra_info("peername")
            if isinstance(peer, tuple) and len(peer) >= 2:
                client = (peer[0], peer[1])

        request = HTTPRequest(
            method=parsed.method or "",
            target=target,
            headers=headers_to_headers(parsed.headers),
            body=_parsed_body(parsed),
        )
        return cls(
            request=request,
            as_received=request,
            wire_raw_bytes=wire_raw_bytes,
            http_version=parsed.http_version or "",
            client=client,
        )

    @property
    def method(self) -> str:
        return self.request.method

    @property
    def target(self) -> str:
        return self.request.target

    @property
    def headers(self) -> Headers:
        return self.request.headers

    @property
    def body(self) -> bytes | None:
        return self.request.body

    @property
    def text(self) -> str:
        return self.request.text

    @property
    def json_body(self) -> Any:
        return self.request.json_body

    @property
    def is_proxy_request(self) -> bool:
        return self.request.is_proxy_request

    @property
    def target_uri(self) -> ParsedURI | None:
        return self.request.target_uri

    @property
    def effective_path(self) -> str:
        return self.request.effective_path

    def with_method(self, method: str) -> RecordedHTTPRequest:
        return replace(self, request=replace(self.request, method=method))

    def with_target(self, target: str) -> RecordedHTTPRequest:
        return replace(self, request=replace(self.request, target=target))

    def with_headers(self, headers: Headers) -> RecordedHTTPRequest:
        return replace(self, request=replace(self.request, headers=headers))

    @property
    def wire_body_bytes(self) -> bytes:
        """Return request body bytes as they appeared on the wire.

        For chunked uploads this includes the original chunk
        framing and any trailer bytes.
        """
        header_end = self.wire_raw_bytes.find(b"\r\n\r\n")
        if header_end == -1:
            return self.as_received.body or b""

        return self.wire_raw_bytes[header_end + 4 :]


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

    def on_chunk_complete(self) -> None:
        """Called at the end of a chunk (for chunked encoding)."""


def _build_request_parser() -> tuple[
    RequestProtocol, httptools.HttpRequestParser
]:
    protocol = RequestProtocol()
    parser = httptools.HttpRequestParser(protocol)
    protocol.set_parser(parser)
    return protocol, parser


class AsyncRequestParser:
    """Async wrapper for httptools.HttpRequestParser with wire tracking.

    This class combines asyncio stream reading with httptools parsing while
    preserving the exact bytes received on the wire. It correctly handles
    pipelined requests by returning bytes read past the end of the
    current request to the stream via ``localstub.http.stream``, where
    any later consumer of the reader picks them up in order.
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
            preserving exact wire bytes. Any bytes belonging to subsequent
            pipelined requests are returned to the stream with
            ``localstub.http.stream.unread_data()`` for the next consumer
            of the reader.
        """
        parsed, wire_bytes, remaining = await self.parse_headers(
            reader,
            connection_wire,
        )
        if parsed is None:
            return None, wire_bytes

        if parsed.is_complete:
            stream.unread_data(reader, remaining)
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
                stream.unread_data(reader, buffer[error_offset:])
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
            stream.unread_data(reader, buffer)
            self._sync_connection_wire(connection_wire)
            return self._protocol.result, bytes(self._wire)

        if is_chunked_transfer(self._protocol.result.headers):
            return await self._parse_chunked_body(
                reader,
                buffer,
                connection_wire,
            )

        body_length = content_length(self._protocol.result.headers)
        if body_length is not None:
            return await self._parse_content_length_body(
                reader,
                buffer,
                body_length,
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

        stream.unread_data(reader, buffer)
        self._sync_connection_wire(connection_wire)
        return self._protocol.result, bytes(self._wire)

    async def _read_more(
        self,
        reader: asyncio.StreamReader,
        buffer: bytearray,
    ) -> bool:
        try:
            data = await stream.read(reader, self._max_read)
        except Exception:
            LOG.debug("Failed to read request data", exc_info=True)
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
        stream.unread_data(reader, buffer[body_offset:])
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
        stream.unread_data(reader, buffer[content_length:])
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
        scan_from = 0
        while True:
            try:
                scan = scan_chunked_body(buffer, scan_from)
            except ChunkScanError as exc:
                self._wire.extend(buffer[: exc.offset])
                self._sync_connection_wire(connection_wire)
                stream.unread_data(reader, buffer[exc.offset :])
                return None, bytes(self._wire)

            if scan.end is not None:
                chunked_end = scan.end
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
                stream.unread_data(reader, buffer[chunked_end:])
                self._sync_connection_wire(connection_wire)

                if not self._protocol.result.is_complete:
                    return None, bytes(self._wire)

                return self._protocol.result, bytes(self._wire)

            scan_from = scan.resume_from
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
    ) -> RecordedHTTPRequest | None:
        """Parse a complete HTTP request from the stream.

        Args:
            reader: The asyncio stream to read from
            writer: Optional writer to extract client info from
            connection_wire: Optional buffer for connection-level tracking

        Returns:
            RecordedHTTPRequest or None on EOF/parse error
        """
        parser = AsyncRequestParser(max_read=self._max_read)
        parsed, wire_bytes = await parser.parse(reader, connection_wire)

        if parsed is None:
            return None

        return RecordedHTTPRequest.from_parsed(
            parsed, wire_bytes, writer=writer
        )
