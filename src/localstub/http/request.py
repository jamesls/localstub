from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, replace
from enum import Enum, auto
from typing import Any, Protocol

import httptools

from localstub.http import stream
from localstub.http.framing import (
    HEADER_TERMINATOR,
    ChunkPayloadScan,
    ChunkScanError,
    content_length,
    is_chunked_transfer,
    scan_chunk_payloads,
)
from localstub.http.headers import HeaderItem, Headers
from localstub.http.stream import ByteStream
from localstub.http.uri import ParsedURI, parse_absolute_uri
from localstub.http.utils import headers_to_headers

LOG = logging.getLogger(__name__)


class Writer(Protocol):
    """Minimal StreamWriter interface for extracting client info."""

    def get_extra_info(self, name: str, default: Any | None = None) -> Any: ...


class WireSink(Protocol):
    """Anything that can accumulate raw wire bytes.

    Satisfied by ``bytearray`` and by bounded buffers that own their
    retention policy.  The parser only ever appends to it.
    """

    def extend(self, data: bytes | bytearray, /) -> None: ...


def client_address(writer: Writer) -> tuple[str, int] | None:
    """Extract the client ``(host, port)`` from a writer's peername."""
    peer: tuple[object, ...] | str | None = writer.get_extra_info("peername")
    match peer:
        case (str() as host, int() as port, *_):
            return (host, port)
        case _:
            return None


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

    ``body_complete`` is ``False`` when the request was recorded before
    its body reached the message boundary: the server stopped reading
    it, the client disconnected, or the body failed to parse.  The
    ``body`` is then the payload prefix consumed before the close and
    ``wire_raw_bytes`` the wire prefix including framing.  It is a fact
    about the observation, not the message, so it lives here and not on
    ``HTTPRequest``.
    """

    request: HTTPRequest
    as_received: HTTPRequest
    wire_raw_bytes: bytes
    http_version: str
    client: tuple[str, int] | None = None
    body_complete: bool = True

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
            client = client_address(writer)

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
            body_complete=parsed.is_complete,
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
    headers: list[tuple[bytes, bytes]] = field(
        default_factory=list[tuple[bytes, bytes]]
    )
    body_parts: list[bytes] = field(default_factory=list[bytes])
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


class ParseStop(Enum):
    """Why a parse call returned."""

    COMPLETE = auto()
    """The requested parse stage finished; check ``parsed.is_complete``
    to learn whether the whole message reached its boundary."""

    EOF = auto()
    """The stream ended before the stage finished."""

    PARSE_ERROR = auto()
    """The bytes could not be parsed as HTTP."""

    READ_ERROR = auto()
    """Reading the stream raised; ``error`` carries the exception."""


@dataclass(frozen=True)
class ParseOutcome:
    """Result of a request parse call.

    ``parsed`` is ``None`` until the headers complete.  After that it
    is present for every stop kind, with the body parts consumed so
    far, so a caller can record a partial request.  ``wire_bytes`` is
    exact for every stop kind: everything attributed to this request,
    including framing, up to the point the parse stopped.
    """

    parsed: ParsedRequest | None
    wire_bytes: bytes
    stop: ParseStop
    error: Exception | None = None

    @property
    def complete_request(self) -> ParsedRequest | None:
        """The parsed request when the whole message reached its boundary."""
        if self.parsed is None or self.stop is not ParseStop.COMPLETE:
            return None
        if not self.parsed.is_complete:
            return None
        return self.parsed


class _ReadStatus(Enum):
    DATA = auto()
    EOF = auto()
    ERROR = auto()


def _chunk_limit_offset(
    scan: ChunkPayloadScan,
    scan_from: int,
    budget: int,
) -> int | None:
    """Map a remaining payload budget to the wire offset to stop at.

    Returns the offset just past the last allowed payload byte and its
    framing. A partial chunk stops as soon as its payload reaches the
    budget. A budget spent exactly at a chunk boundary stops there as
    soon as the scan shows another chunk follows, even before any of
    that chunk's data arrives. Otherwise, ``None`` means the caller
    needs the message boundary or more data to decide.
    """
    spans = [*scan.payloads]
    if scan.partial is not None:
        spans.append(scan.partial)
    if budget == 0:
        return scan_from if spans else None

    boundary = scan_from
    for index, (start, end) in enumerate(scan.payloads):
        size = end - start
        if size > budget:
            return start + budget
        budget -= size
        boundary = end + 2
        if budget == 0:
            return boundary if len(spans) > index + 1 else None

    if scan.partial is not None:
        start, end = scan.partial
        if end - start >= budget:
            return start + budget
    return None


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
        self._upgraded = False
        self._read_error: Exception | None = None
        self._header_error: Exception | None = None

    async def parse(
        self,
        reader: ByteStream,
        connection_wire: WireSink | None = None,
    ) -> ParseOutcome:
        """Parse a complete HTTP request from the stream.

        Args:
            reader: The stream to read from.
            connection_wire: Optional buffer to accumulate connection-level
                bytes (for tracking across multiple requests).

        Returns:
            The parse outcome; ``complete_request`` is the request when
            it reached its message boundary.

        Note:
            This method parses headers and bodies in buffered segments while
            preserving exact wire bytes. Any bytes belonging to subsequent
            pipelined requests are returned to the stream with
            ``localstub.http.stream.unread_data()`` for the next consumer
            of the reader.
        """
        outcome, remaining = await self.parse_headers(
            reader,
            connection_wire,
        )
        parsed = outcome.parsed
        if parsed is None or outcome.stop is not ParseStop.COMPLETE:
            return outcome

        if parsed.is_complete:
            stream.unread_data(reader, remaining)
            return outcome

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
        reader: ByteStream,
        connection_wire: WireSink | None = None,
    ) -> tuple[ParseOutcome, bytearray]:
        """Parse request headers only, stopping before body.

        Args:
            reader: The stream to read from.
            connection_wire: Optional buffer to accumulate connection-level
                bytes (for tracking across multiple requests).

        Returns:
            Tuple of (outcome, remaining_buffer).  The outcome's
            ``parsed`` is present once the headers are complete, and
            ``remaining_buffer`` holds bytes read but not yet processed,
            which should be passed to continue_parse_body().  On any
            other stop the remaining buffer is empty.  A ``PARSE_ERROR``
            stop can carry ``parsed``: the parser rejects some requests
            at the header boundary, after the headers completed, such
            as a ``Transfer-Encoding`` whose final coding is not
            ``chunked`` (RFC 9112 §6.1).  Check ``stop`` before reading
            the body.
        """
        buffer = bytearray()
        fed = 0
        status = _ReadStatus.DATA

        while not self._protocol.result.headers_complete:
            header_end = buffer.find(HEADER_TERMINATOR)
            if header_end == -1:
                status = await self._read_more(reader, buffer)
                if status is not _ReadStatus.DATA:
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
            except httptools.HttpParserUpgrade:
                self._note_upgrade()
            except httptools.HttpParserError as exc:
                self._header_error = exc
                error_offset = self._precise_error_offset(
                    bytes(buffer[:feed_end])
                )
                self._wire.extend(buffer[:error_offset])
                self._sync_connection_wire(connection_wire)
                stream.unread_data(reader, buffer[error_offset:])
                return self._outcome(ParseStop.PARSE_ERROR, exc), bytearray()

            fed = feed_end

        if not self._protocol.result.headers_complete:
            self._wire.extend(buffer)
            self._sync_connection_wire(connection_wire)
            return self._headers_stopped(status), bytearray()

        header_end = buffer.find(HEADER_TERMINATOR) + len(HEADER_TERMINATOR)
        self._wire.extend(buffer[:header_end])
        self._sync_connection_wire(connection_wire)
        return self._outcome(ParseStop.COMPLETE), buffer[header_end:]

    async def continue_parse_body(
        self,
        reader: ByteStream,
        remaining_buffer: bytearray,
        connection_wire: WireSink | None = None,
        *,
        max_body_bytes: int | None = None,
    ) -> ParseOutcome:
        """Continue parsing the request body after headers.

        Call this after parse_headers() and after sending any interim
        response (like 100 Continue).

        Args:
            reader: The stream to continue reading from.
            remaining_buffer: Buffer returned from parse_headers().
            connection_wire: Optional buffer for connection-level tracking.
            max_body_bytes: Stop after consuming this many payload bytes,
                excluding transfer framing, or at the message boundary,
                whichever comes first.  ``None`` reads to the boundary.

        Returns:
            The parse outcome.  Its ``wire_bytes`` covers the whole
            request read so far; bytes beyond the stopping point are
            returned to the stream.
        """
        buffer = remaining_buffer
        result = self._protocol.result

        if result.is_complete or max_body_bytes == 0:
            stream.unread_data(reader, buffer)
            self._sync_connection_wire(connection_wire)
            return self._outcome(ParseStop.COMPLETE)

        if is_chunked_transfer(result.headers):
            return await self._parse_chunked_body(
                reader,
                buffer,
                connection_wire,
                max_body_bytes,
            )

        body_length = content_length(result.headers)
        if body_length is None:
            return self._unframed_body(reader, buffer, connection_wire)
        return await self._parse_content_length_body(
            reader,
            buffer,
            body_length,
            connection_wire,
            max_body_bytes,
        )

    def _unframed_body(
        self,
        reader: ByteStream,
        buffer: bytearray,
        connection_wire: WireSink | None,
    ) -> ParseOutcome:
        """Refuse to read a body whose length the headers do not give.

        httptools leaves a request open past its headers only when
        they framed a body, so reaching here means it rejected the
        request at the header boundary and ``parse_headers()`` already
        reported that.  The stage stays a parse error rather than
        reading an undelimited body; the buffered bytes go back to the
        stream.
        """
        stream.unread_data(reader, buffer)
        self._sync_connection_wire(connection_wire)
        return self._outcome(ParseStop.PARSE_ERROR, self._header_error)

    def snapshot(self) -> ParseOutcome:
        """Describe what has been parsed so far without reading further.

        For a parse call that was cancelled: the outcome carries the
        headers and the body parts delivered before the interruption,
        as a ``COMPLETE`` stop whose message is incomplete, the same
        shape as a stage the caller limited on purpose.
        """
        return self._outcome(ParseStop.COMPLETE)

    def _outcome(
        self,
        stop: ParseStop,
        error: Exception | None = None,
    ) -> ParseOutcome:
        result = self._protocol.result
        parsed = result if result.headers_complete else None
        return ParseOutcome(parsed, bytes(self._wire), stop, error)

    def _headers_stopped(self, status: _ReadStatus) -> ParseOutcome:
        """Outcome for a header read that ended before they completed.

        With data still flowing the loop only stops when the parser
        refused to complete the headers, which is a parse failure.
        """
        if status is _ReadStatus.DATA:
            return self._outcome(ParseStop.PARSE_ERROR)
        return self._body_outcome(status)

    def _body_outcome(self, status: _ReadStatus) -> ParseOutcome:
        """Outcome for a body read; the stage completed unless the
        stream ended or failed first."""
        if status is _ReadStatus.ERROR:
            return self._outcome(ParseStop.READ_ERROR, self._read_error)
        if status is _ReadStatus.EOF:
            return self._outcome(ParseStop.EOF)
        return self._outcome(ParseStop.COMPLETE)

    async def _read_more(
        self,
        reader: ByteStream,
        buffer: bytearray,
    ) -> _ReadStatus:
        try:
            data = await stream.read(reader, self._max_read)
        except Exception as exc:
            LOG.debug("Failed to read request data", exc_info=True)
            self._read_error = exc
            return _ReadStatus.ERROR

        if not data:
            return _ReadStatus.EOF

        buffer.extend(data)
        return _ReadStatus.DATA

    def _sync_connection_wire(
        self,
        connection_wire: WireSink | None,
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
        reader: ByteStream,
        buffer: bytearray,
        consumed_guess: int,
        connection_wire: WireSink | None,
        error: Exception,
    ) -> ParseOutcome:
        candidate = bytes(self._wire) + bytes(buffer[:consumed_guess])
        error_offset = self._precise_error_offset(candidate)
        body_offset = max(0, error_offset - len(self._wire))
        self._wire.extend(buffer[:body_offset])
        self._sync_connection_wire(connection_wire)
        stream.unread_data(reader, buffer[body_offset:])
        return self._outcome(ParseStop.PARSE_ERROR, error)

    def _note_upgrade(self) -> None:
        """Arrange for an Upgrade request's declared body to be read.

        httptools stops at the header boundary of an Upgrade request
        and marks the message complete without parsing any declared
        body (RFC 9110 allows Upgrade requests to carry one).  Clear
        the premature completion flag so the body framing logic runs;
        the body is then consumed without the parser, which accepts
        no further data after the upgrade.  CONNECT requests have no
        content, so their tunnel bytes are never mistaken for a body,
        and ``Content-Length: 0`` declares none, so the header boundary
        stays the message boundary.
        """
        self._upgraded = True
        result = self._protocol.result
        if result.method == "CONNECT":
            return
        declared = content_length(result.headers)
        if is_chunked_transfer(result.headers) or (
            declared is not None and declared > 0
        ):
            result.is_complete = False

    def _deliver_upgrade_body(
        self,
        parts: list[bytes],
        *,
        complete: bool,
    ) -> None:
        """Record an Upgrade request's body via the protocol callbacks.

        The httptools parser refuses data after an upgrade, so the
        callbacks it would have made for the body are made directly.
        """
        for part in parts:
            self._protocol.on_body(part)
        if complete:
            self._protocol.on_message_complete()

    async def _parse_content_length_body(
        self,
        reader: ByteStream,
        buffer: bytearray,
        content_length: int,
        connection_wire: WireSink | None,
        max_body_bytes: int | None,
    ) -> ParseOutcome:
        target = (
            content_length
            if max_body_bytes is None
            else min(max_body_bytes, content_length)
        )
        consumed = 0
        status = _ReadStatus.DATA
        while True:
            take = min(len(buffer), target - consumed)
            if take:
                # Body bytes under a Content-Length cannot fail to parse.
                self._feed_body(buffer, take)
                consumed += take
                self._sync_connection_wire(connection_wire)
            if consumed >= target:
                break
            status = await self._read_more(reader, buffer)
            if status is not _ReadStatus.DATA:
                break

        if self._upgraded and consumed == content_length:
            self._protocol.on_message_complete()
        stream.unread_data(reader, buffer)
        self._sync_connection_wire(connection_wire)
        return self._body_outcome(status)

    def _feed_body(
        self,
        buffer: bytearray,
        length: int,
        *,
        chunked: bool = False,
    ) -> None:
        """Deliver ``buffer[:length]`` as body bytes and attribute them.

        The bytes are removed from the buffer once the parser accepted
        them; when the parser raises they stay so the error offset can
        be located.  ``chunked`` bytes are complete chunks whose
        framing is stripped when the parser cannot be fed after an
        upgrade.
        """
        segment = bytes(buffer[:length])
        if self._upgraded:
            parts = (
                _chunk_payload_parts(buffer[:length]) if chunked else [segment]
            )
            self._deliver_upgrade_body(parts, complete=False)
        else:
            self._parser.feed_data(segment)
        self._wire.extend(segment)
        del buffer[:length]

    async def _parse_chunked_body(
        self,
        reader: ByteStream,
        buffer: bytearray,
        connection_wire: WireSink | None,
        max_body_bytes: int | None,
    ) -> ParseOutcome:
        """Consume chunk framing, feeding complete chunks as they arrive.

        ``buffer`` only ever holds bytes not yet delivered to the
        parser, so every scan starts at a chunk-size boundary at offset
        zero and a cancelled parse leaves the delivered prefix recorded.
        """
        counted = 0
        while True:
            try:
                scan = scan_chunk_payloads(buffer)
            except ChunkScanError as exc:
                return self._chunk_scan_error(
                    reader,
                    buffer,
                    exc,
                    connection_wire,
                    budget=(
                        None
                        if max_body_bytes is None
                        else max_body_bytes - counted
                    ),
                )

            if max_body_bytes is not None:
                stop_at = _chunk_limit_offset(
                    scan, 0, max_body_bytes - counted
                )
                if stop_at is not None:
                    return self._finish_chunked_prefix(
                        reader,
                        buffer,
                        stop_at,
                        connection_wire,
                        _ReadStatus.DATA,
                        complete=False,
                    )

            if scan.end is not None:
                return self._finish_chunked_prefix(
                    reader,
                    buffer,
                    scan.end,
                    connection_wire,
                    _ReadStatus.DATA,
                    complete=True,
                )

            counted += sum(end - start for start, end in scan.payloads)
            if scan.resume_from:
                try:
                    self._feed_body(buffer, scan.resume_from, chunked=True)
                except httptools.HttpParserError as exc:
                    return self._body_parse_error(
                        reader,
                        buffer,
                        scan.resume_from,
                        connection_wire,
                        exc,
                    )
                self._sync_connection_wire(connection_wire)
            status = await self._read_more(reader, buffer)
            if status is not _ReadStatus.DATA:
                return self._finish_chunked_prefix(
                    reader,
                    buffer,
                    len(buffer),
                    connection_wire,
                    status,
                    complete=False,
                )

    def _chunk_scan_error(
        self,
        reader: ByteStream,
        buffer: bytearray,
        error: ChunkScanError,
        connection_wire: WireSink | None,
        *,
        budget: int | None,
    ) -> ParseOutcome:
        """Attribute the bytes up to a bad chunk-framing byte to the request.

        The payload before the bad byte is still delivered, to the
        parser or, after an upgrade, through the upgrade-body callbacks,
        so the recorded body reflects what was consumed before the
        error however the reads were split.
        """
        if budget is not None:
            # The scanner may have reached an error beyond the payload
            # limit. Rescan only the valid prefix before attributing it.
            scan = scan_chunk_payloads(buffer[: error.offset - 1])
            stop_at = _chunk_limit_offset(scan, 0, budget)
            if (
                stop_at is None
                and sum(end - start for start, end in scan.payloads) == budget
            ):
                stop_at = scan.resume_from
            if stop_at is not None:
                return self._finish_chunked_prefix(
                    reader,
                    buffer,
                    stop_at,
                    connection_wire,
                    _ReadStatus.DATA,
                    complete=False,
                )

        prefix = bytes(buffer[: error.offset])
        if self._upgraded:
            self._deliver_upgrade_body(
                _chunk_payload_parts(buffer[: error.offset - 1]),
                complete=False,
            )
        else:
            try:
                self._parser.feed_data(prefix)
            except httptools.HttpParserError:
                LOG.debug("Parser rejected chunk prefix", exc_info=True)
        self._wire.extend(prefix)
        self._sync_connection_wire(connection_wire)
        stream.unread_data(reader, buffer[error.offset :])
        return self._outcome(ParseStop.PARSE_ERROR, error)

    def _finish_chunked_prefix(
        self,
        reader: ByteStream,
        buffer: bytearray,
        offset: int,
        connection_wire: WireSink | None,
        status: _ReadStatus,
        *,
        complete: bool,
    ) -> ParseOutcome:
        """Consume the undelivered chunked bytes up to ``offset``.

        ``offset`` is the message boundary when the terminal chunk was
        reached, the threshold stop inside the body, or the end of the
        buffer when the stream ended first.
        """
        if self._upgraded:
            self._deliver_upgrade_body(
                _chunk_payload_parts(buffer[:offset]), complete=complete
            )
            self._wire.extend(buffer[:offset])
        else:
            try:
                self._parser.feed_data(bytes(buffer[:offset]))
            except httptools.HttpParserError as exc:
                return self._body_parse_error(
                    reader, buffer, offset, connection_wire, exc
                )
            self._wire.extend(buffer[:offset])

        stream.unread_data(reader, buffer[offset:])
        self._sync_connection_wire(connection_wire)
        return self._body_outcome(status)


def _chunk_payload_parts(buffer: bytearray) -> list[bytes]:
    """Extract chunk payloads, including a trailing partial chunk.

    A partial chunk whose data has not started yet contributes no
    part, so a bare size line never records an empty body part.
    """
    scan = scan_chunk_payloads(buffer)
    parts = [bytes(buffer[start:end]) for start, end in scan.payloads]
    if scan.partial is not None:
        start, end = scan.partial
        if end > start:
            parts.append(bytes(buffer[start:end]))
    return parts


class HTTPRequestReader:
    """Public interface for reading HTTP requests from streams.

    This provides a clean way to parse HTTP requests without needing
    an AsyncHTTPTestServer instance.
    """

    def __init__(self, max_read: int = 8192) -> None:
        self._max_read = max_read

    async def read_request(
        self,
        reader: ByteStream,
        writer: Writer | None = None,
        connection_wire: WireSink | None = None,
    ) -> RecordedHTTPRequest | None:
        """Parse a complete HTTP request from the stream.

        Args:
            reader: The stream to read from
            writer: Optional writer to extract client info from
            connection_wire: Optional buffer for connection-level tracking

        Returns:
            RecordedHTTPRequest or None on EOF/parse error, or when the
            request did not reach its message boundary
        """
        parser = AsyncRequestParser(max_read=self._max_read)
        outcome = await parser.parse(reader, connection_wire)

        parsed = outcome.complete_request
        if parsed is None:
            return None

        return RecordedHTTPRequest.from_parsed(
            parsed, outcome.wire_bytes, writer=writer
        )
