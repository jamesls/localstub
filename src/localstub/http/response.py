from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from email.message import Message

import httptools

from localstub.http.framing import (
    HEADER_TERMINATOR,
    ChunkScanError,
    content_length,
    is_chunked_transfer,
    scan_chunked_body,
)


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

    Parses framed segments while precisely tracking where each response
    ends, preserving exact wire bytes for each response.
    """

    def __init__(self, max_read: int = 8192) -> None:
        self._max_read = max_read
        self._buffer = bytearray()

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
        header_end = await self._read_headers(reader)
        if header_end is None:
            wire = bytes(self._buffer)
            self._buffer.clear()
            return None, wire

        header_wire = bytes(self._buffer[:header_end])
        try:
            parser.feed_data(header_wire)
        except httptools.HttpParserError:
            error_offset = self._precise_error_offset(header_wire)
            wire = bytes(self._buffer[:error_offset])
            del self._buffer[:error_offset]
            return None, wire

        del self._buffer[:header_end]
        wire = bytearray(header_wire)

        if protocol.message_complete:
            return protocol.result, bytes(wire)

        if is_head and protocol.result.http_version is not None:
            protocol.result.is_complete = True
            return protocol.result, bytes(wire)

        if is_chunked_transfer(protocol.result.headers):
            return await self._parse_chunked_body(
                reader,
                protocol,
                parser,
                wire,
            )

        body_length = content_length(protocol.result.headers)
        if body_length is not None:
            return await self._parse_content_length_body(
                reader,
                protocol,
                parser,
                wire,
                body_length,
            )

        return await self._parse_close_delimited_body(
            reader,
            protocol,
            parser,
            wire,
        )

    async def _read_more(self, reader: asyncio.StreamReader) -> bool:
        try:
            data = await reader.read(self._max_read)
        except Exception:
            return False

        if not data:
            return False

        self._buffer.extend(data)
        return True

    async def _read_headers(
        self,
        reader: asyncio.StreamReader,
    ) -> int | None:
        while True:
            header_end = self._buffer.find(HEADER_TERMINATOR)
            if header_end != -1:
                return header_end + len(HEADER_TERMINATOR)
            if not await self._read_more(reader):
                return None

    def _precise_error_offset(self, data: bytes) -> int:
        protocol = ResponseProtocol()
        parser = httptools.HttpResponseParser(protocol)
        protocol.set_parser(parser)

        for index, value in enumerate(data, start=1):
            try:
                parser.feed_data(bytes((value,)))
            except httptools.HttpParserError:
                return index

            if protocol.message_complete:
                return index

        return len(data)

    async def _parse_content_length_body(
        self,
        reader: asyncio.StreamReader,
        protocol: ResponseProtocol,
        parser: httptools.HttpResponseParser,
        wire: bytearray,
        content_length: int,
    ) -> tuple[ParsedResponse | None, bytes]:
        remaining = content_length

        while remaining > 0:
            if not self._buffer and not await self._read_more(reader):
                return None, bytes(wire)

            segment_length = min(remaining, len(self._buffer))
            segment = bytes(self._buffer[:segment_length])
            try:
                parser.feed_data(segment)
            except httptools.HttpParserError:
                return self._body_parse_error(wire, segment_length)

            wire.extend(segment)
            del self._buffer[:segment_length]
            remaining -= segment_length

        if not protocol.message_complete:
            return None, bytes(wire)
        return protocol.result, bytes(wire)

    async def _parse_chunked_body(
        self,
        reader: asyncio.StreamReader,
        protocol: ResponseProtocol,
        parser: httptools.HttpResponseParser,
        wire: bytearray,
    ) -> tuple[ParsedResponse | None, bytes]:
        scan_from = 0
        while True:
            try:
                scan = scan_chunked_body(self._buffer, scan_from)
            except ChunkScanError as exc:
                wire.extend(self._buffer[: exc.offset])
                del self._buffer[: exc.offset]
                return None, bytes(wire)

            if scan.end is not None:
                chunked_end = scan.end
                chunked_body = bytes(self._buffer[:chunked_end])
                try:
                    parser.feed_data(chunked_body)
                except httptools.HttpParserError:
                    return self._body_parse_error(wire, chunked_end)

                wire.extend(chunked_body)
                del self._buffer[:chunked_end]
                if not protocol.message_complete:
                    return None, bytes(wire)
                return protocol.result, bytes(wire)

            scan_from = scan.resume_from
            if not await self._read_more(reader):
                wire.extend(self._buffer)
                self._buffer.clear()
                return None, bytes(wire)

    async def _parse_close_delimited_body(
        self,
        reader: asyncio.StreamReader,
        protocol: ResponseProtocol,
        parser: httptools.HttpResponseParser,
        wire: bytearray,
    ) -> tuple[ParsedResponse | None, bytes]:
        while True:
            if self._buffer:
                segment = bytes(self._buffer)
                try:
                    parser.feed_data(segment)
                except httptools.HttpParserError:
                    return self._body_parse_error(wire, len(segment))
                wire.extend(segment)
                self._buffer.clear()

            if not await self._read_more(reader):
                break

        if protocol.result.http_version is not None and _is_close_delimited(
            protocol.result.headers
        ):
            protocol.result.is_complete = True
            return protocol.result, bytes(wire)
        return None, bytes(wire)

    def _body_parse_error(
        self,
        wire: bytearray,
        consumed_guess: int,
    ) -> tuple[ParsedResponse | None, bytes]:
        candidate = bytes(wire) + bytes(self._buffer[:consumed_guess])
        error_offset = self._precise_error_offset(candidate)
        body_offset = max(0, error_offset - len(wire))
        wire.extend(self._buffer[:body_offset])
        del self._buffer[:body_offset]
        return None, bytes(wire)
