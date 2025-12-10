import asyncio
from dataclasses import dataclass, field

import httptools


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
        request_method: str | None = None,
    ) -> tuple[ParsedResponse | None, bytes]:
        """Parse a complete HTTP response from the stream.

        Args:
            reader: The asyncio stream to read from.
            request_method: The HTTP method of the request that generated
                this response. Required for HEAD requests, which have no
                body despite Content-Length headers.

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

            # HEAD responses have no body - complete once headers are done
            if is_head and self._protocol.result.http_version is not None:
                self._protocol.result.is_complete = True
                break

        # For close-delimited bodies, the message may be complete after EOF
        # even if on_message_complete wasn't called
        if not self._protocol.result.is_complete:
            # Only treat as close-delimited if we have headers AND there's
            # no Content-Length or Transfer-Encoding header. If those headers
            # exist, the response is truncated and should be treated as a
            # parse failure.
            if self._protocol.result.http_version is not None:
                if self._is_close_delimited():
                    self._protocol.result.is_complete = True
                else:
                    # Truncated response - Content-Length or chunked encoding
                    # indicated more data was expected
                    return None, bytes(self._wire)
            else:
                return None, bytes(self._wire)

        return self._protocol.result, bytes(self._wire)

    def _is_close_delimited(self) -> bool:
        """Check if the response is genuinely close-delimited.

        A response is close-delimited if it has no Content-Length header
        and no Transfer-Encoding header. In this case, EOF signals the
        end of the response body.

        Returns:
            True if the response is close-delimited, False otherwise.
        """
        headers = self._protocol.result.headers
        for name, _ in headers:
            name_lower = name.lower()
            if name_lower == b"content-length":
                return False
            if name_lower == b"transfer-encoding":
                return False
        return True

    @property
    def wire_bytes(self) -> bytes:
        """Return the accumulated wire bytes."""
        return bytes(self._wire)
