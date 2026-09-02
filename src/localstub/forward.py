"""Shared HTTP forwarding logic for proxy modes.

This module provides the RawForwarder class which handles forwarding
HTTP requests to upstream servers and relaying responses back to
clients, preserving exact wire bytes including Transfer-Encoding.
"""

from __future__ import annotations

import asyncio
import gzip
import logging
import ssl
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from email.message import Message
from enum import Enum
from typing import Protocol

import truststore

from localstub.http.connection import should_close_connection
from localstub.http.headers import Headers
from localstub.http.request import (
    AsyncRequestParser,
    ParsedRequest,
    RecordedHTTPRequest,
)
from localstub.http.response import (
    AsyncMultiResponseParser,
    ParsedResponse,
    RecordedHTTPResponse,
)
from localstub.http.responsespec import HTTPResponse
from localstub.http.utils import (
    decode_status_text,
    headers_to_message,
    maybe_await,
    serialize_header_line,
    status_phrase,
)

LOG = logging.getLogger(__name__)

WireLog = Callable[[str, bytes], None]


def upstream_ssl_context(verify: bool) -> ssl.SSLContext:
    """Build the TLS context used for upstream connections.

    Verified contexts use system trust (truststore); unverified
    contexts disable certificate and hostname checks entirely.
    """
    if verify:
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    return ssl_ctx


async def open_upstream_connection(
    host: str,
    port: int,
    *,
    use_tls: bool,
    verify: bool = True,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a TCP (optionally TLS) stream connection to *host*:*port*.

    Raises OSError (including ssl.SSLError) on connection failure.
    """
    ssl_ctx = upstream_ssl_context(verify) if use_tls else None
    server_hostname = host if use_tls else None
    return await asyncio.open_connection(
        host, port, ssl=ssl_ctx, server_hostname=server_hostname
    )


class ClientWriter(Protocol):
    """Minimal writer interface required for relaying responses."""

    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...

    def close(self) -> None: ...

    async def wait_closed(self) -> None: ...


@dataclass
class TransformContext:
    """Upstream response handed to a ResponseTransformer."""

    status: int
    reason: str | None
    headers: Message
    body: bytes
    wire_raw_bytes: bytes


@dataclass
class TransformResult:
    """Result of transforming an upstream response."""

    body: bytes | None = None
    override_response: HTTPResponse | None = None
    delay_before: float = 0.0
    drop_after: int | None = None


ResponseTransformer = Callable[
    [TransformContext],
    TransformResult | Awaitable[TransformResult],
]


@dataclass(frozen=True)
class RelayedResponse:
    """Connection-relevant metadata of the response sent to the client.

    A transformer may rewrite or override the upstream message, so the
    response the client saw can differ from the recorded upstream
    response; connection-persistence decisions must follow the relayed
    message rather than the upstream one.
    """

    status: int
    headers: Headers | Message
    http_version: str | None
    is_eof_delimited: bool


@dataclass
class ForwardResult:
    """Result of forwarding a request to upstream.

    The ``status``/``reason``/``headers``/``body``/``wire_bytes``
    fields describe the upstream message as received (what gets
    recorded); ``relayed`` describes the response actually sent to
    the client.
    """

    status: int
    reason: str | None
    headers: Message
    body: bytes
    wire_bytes: bytes
    relayed: RelayedResponse
    is_eof_delimited: bool = False

    def to_recorded_response(
        self,
        wire_raw_bytes: bytes | None = None,
    ) -> RecordedHTTPResponse:
        """Convert to a RecordedHTTPResponse for recording."""
        return RecordedHTTPResponse(
            response=HTTPResponse(
                status=self.status,
                headers=Headers.from_items(self.headers.items()),
                body=self.body,
            ),
            reason=self.reason,
            wire_raw_bytes=(
                wire_raw_bytes
                if wire_raw_bytes is not None
                else self.wire_bytes
            ),
        )


def response_allows_keep_alive(
    request: RecordedHTTPRequest,
    result: ForwardResult,
) -> bool:
    """Return whether the client connection may carry another request.

    The decision follows the response actually relayed to the client
    (``result.relayed``).  Reuse is ruled out by a protocol switch,
    an EOF-delimited body (the client learns completion from the
    close), a ``Connection: close`` from either party, or a relayed
    HTTP/1.0 response that did not opt in to keep-alive.
    """
    relayed = result.relayed
    if relayed.status == 101:
        # The connection now speaks the upgraded protocol; without a
        # bidirectional relay it cannot carry further HTTP.
        return False
    if relayed.is_eof_delimited:
        return False
    return not should_close_connection(
        request,
        response_headers=relayed.headers,
        response_version=relayed.http_version,
    )


class ForwardError(Enum):
    """Forwarding error category."""

    REQUEST_PARSE_FAILED = "request_parse_failed"
    RESPONSE_PARSE_FAILED = "response_parse_failed"


@dataclass
class ForwardedRequest:
    """Result of forwarding a request that originated from a client.

    ``request_body_consumed`` reports whether the client's request body
    was fully read.  When false, unread body bytes may still arrive on
    the connection, so it cannot carry another HTTP request.
    """

    parsed_request: ParsedRequest
    request_wire_bytes: bytes
    response: ForwardResult | None
    request_body_consumed: bool
    error: ForwardError | None = None


class RawForwarder:
    """Forwards HTTP requests to upstream servers and relays responses.

    This class provides raw socket-based forwarding that preserves exact
    wire bytes, including Transfer-Encoding headers and chunked framing.
    It is a wire-fidelity socket relay; HTTP-level forwarding goes
    through an ``HTTPClient`` instead.
    """

    def __init__(
        self,
        max_read: int = 8192,
        verify_upstream: bool = True,
        response_transformer: ResponseTransformer | None = None,
        decompress_body: bool = False,
        wire_log: WireLog | None = None,
    ) -> None:
        """Initialize the forwarder.

        Args:
            max_read: Maximum bytes to read per chunk.
            verify_upstream: Whether to verify upstream TLS certificates.
            response_transformer: Optional transformer for final responses.
            decompress_body: Whether to transparently decompress gzip bodies.
            wire_log: Optional callback for logging wire bytes.
        """
        self._max_read = max_read
        self._verify_upstream = verify_upstream
        self._response_transformer = response_transformer
        self._decompress_body = decompress_body
        self._wire_log = wire_log

    async def forward_and_relay(
        self,
        host: str,
        port: int,
        request_wire_bytes: bytes,
        client_writer: ClientWriter,
        request_method: str | None = None,
        upstream_tls: bool = False,
        upstream_id: str | None = None,
        client_id: str | None = None,
    ) -> ForwardResult | None:
        """Forward request to upstream and relay response to client.

        Args:
            host: Upstream host to connect to.
            port: Upstream port to connect to.
            request_wire_bytes: Complete HTTP request as wire bytes.
            client_writer: Stream writer to relay response to.
            request_method: HTTP method (needed for HEAD response handling).
            upstream_tls: Whether to use TLS for the upstream connection.
            upstream_id: Optional label for upstream in wire logs.
            client_id: Optional label for client in wire logs.

        Returns:
            ForwardResult for recording, or None on connection failure.
        """
        upstream = await self._connect_upstream(host, port, upstream_tls)
        if upstream is None:
            return None

        upstream_reader, upstream_writer = upstream
        try:
            if self._wire_log is not None:
                upstream_label = upstream_id or f"{host}:{port}"
                self._wire_log(
                    f"{upstream_label} <-- lstub",
                    request_wire_bytes,
                )
            upstream_writer.write(request_wire_bytes)
            await upstream_writer.drain()

            return await self.read_and_relay_responses(
                upstream_reader=upstream_reader,
                client_writer=client_writer,
                request_method=request_method,
                upstream_id=upstream_id,
                client_id=client_id,
            )
        finally:
            upstream_writer.close()
            try:
                await upstream_writer.wait_closed()
            except OSError:
                LOG.debug(
                    "Failed to close upstream writer",
                    exc_info=True,
                )

    async def connect_upstream(
        self,
        host: str,
        port: int,
        *,
        upstream_tls: bool,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
        """Connect to an upstream server."""
        return await self._connect_upstream(host, port, upstream_tls)

    async def _connect_upstream(
        self,
        host: str,
        port: int,
        use_tls: bool,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
        """Connect to upstream server.

        Args:
            host: Host to connect to.
            port: Port to connect to.
            use_tls: Whether to use TLS for the connection.

        Returns:
            Tuple of (reader, writer), or None on failure.
        """
        try:
            return await open_upstream_connection(
                host,
                port,
                use_tls=use_tls,
                verify=self._verify_upstream,
            )
        except Exception as e:
            LOG.warning(
                "Failed to connect to upstream %s:%d: %s",
                host,
                port,
                e,
                exc_info=True,
            )
            return None

    async def read_and_relay_responses(
        self,
        upstream_reader: asyncio.StreamReader,
        client_writer: ClientWriter,
        request_method: str | None,
        *,
        upstream_id: str | None = None,
        client_id: str | None = None,
        multi_parser: AsyncMultiResponseParser | None = None,
    ) -> ForwardResult | None:
        """Read responses from upstream and relay to client.

        Relays informational responses until a final response or a 101
        protocol switch is received.

        Args:
            upstream_reader: Stream reader for upstream connection.
            client_writer: Stream writer for client connection.
            request_method: HTTP method (needed for HEAD response handling).

        Returns:
            ForwardResult for the final response, or None on parse failure.
        """
        if multi_parser is None:
            multi_parser = AsyncMultiResponseParser(max_read=self._max_read)

        while True:
            parsed, wire_bytes = await multi_parser.next_response(
                upstream_reader, request_method
            )
            if parsed is None:
                return None

            status = parsed.status_code or 0

            upstream_label = upstream_id or "upstream"
            client_label = client_id or "client"

            if self._wire_log is not None:
                self._wire_log(f"{upstream_label} --> lstub", wire_bytes)

            is_final = status == 101 or status >= 200

            if self._response_transformer is not None and status >= 200:
                return await self._apply_transformation_and_relay(
                    parsed=parsed,
                    wire_bytes=wire_bytes,
                    client_writer=client_writer,
                    client_label=client_label,
                )

            # Relay wire bytes directly to client
            if self._wire_log is not None:
                self._wire_log(f"lstub --> {client_label}", wire_bytes)
            client_writer.write(wire_bytes)
            await client_writer.drain()

            if is_final:
                return self._build_result(parsed, wire_bytes)

            # For 1xx responses, continue reading for final response
            LOG.debug(
                "Received 1xx informational response (%d), "
                "waiting for final response",
                status,
            )

    async def forward_with_100_continue(
        self,
        *,
        parser: AsyncRequestParser,
        parsed_headers: ParsedRequest,
        header_wire_bytes: bytes,
        remaining_buffer: bytearray,
        client_reader: asyncio.StreamReader,
        client_writer: ClientWriter,
        upstream_reader: asyncio.StreamReader,
        upstream_writer: asyncio.StreamWriter,
        request_method: str | None,
        upstream_id: str | None = None,
        client_id: str | None = None,
    ) -> ForwardedRequest:
        """Forward request when client waits for 100-continue."""
        upstream_label = upstream_id or "upstream"
        client_label = client_id or "client"

        resp_parser = AsyncMultiResponseParser(max_read=self._max_read)

        if self._wire_log is not None:
            self._wire_log(f"{upstream_label} <-- lstub", header_wire_bytes)
        upstream_writer.write(header_wire_bytes)
        await upstream_writer.drain()

        (
            parsed_response,
            response_wire,
        ) = await self._relay_until_continue_or_final(
            parser=resp_parser,
            upstream_reader=upstream_reader,
            client_writer=client_writer,
            request_method=request_method,
            upstream_label=upstream_label,
            client_label=client_label,
        )
        if parsed_response is None:
            return ForwardedRequest(
                parsed_request=parsed_headers,
                request_wire_bytes=(
                    header_wire_bytes + bytes(remaining_buffer)
                ),
                response=None,
                request_body_consumed=False,
                error=ForwardError.RESPONSE_PARSE_FAILED,
            )

        status = parsed_response.status_code or 0
        if status == 100:
            final_parsed, full_wire = await parser.continue_parse_body(
                client_reader, remaining_buffer
            )
            if final_parsed is None:
                return ForwardedRequest(
                    parsed_request=parsed_headers,
                    request_wire_bytes=(
                        header_wire_bytes + bytes(remaining_buffer)
                    ),
                    response=None,
                    request_body_consumed=False,
                    error=ForwardError.REQUEST_PARSE_FAILED,
                )

            body_wire = full_wire[len(header_wire_bytes) :]
            if body_wire:
                if self._wire_log is not None:
                    self._wire_log(f"{upstream_label} <-- lstub", body_wire)
                upstream_writer.write(body_wire)
                await upstream_writer.drain()

            response = await self.read_and_relay_responses(
                upstream_reader=upstream_reader,
                client_writer=client_writer,
                request_method=request_method,
                upstream_id=upstream_id,
                client_id=client_id,
                multi_parser=resp_parser,
            )
            return ForwardedRequest(
                parsed_request=final_parsed,
                request_wire_bytes=full_wire,
                response=response,
                request_body_consumed=True,
                error=(
                    None
                    if response is not None
                    else ForwardError.RESPONSE_PARSE_FAILED
                ),
            )

        response: ForwardResult | None
        if self._response_transformer is not None and status >= 200:
            response = await self._apply_transformation_and_relay(
                parsed=parsed_response,
                wire_bytes=response_wire,
                client_writer=client_writer,
                client_label=client_label,
            )
        else:
            if self._wire_log is not None:
                self._wire_log(f"{upstream_label} --> lstub", response_wire)
                self._wire_log(f"lstub --> {client_label}", response_wire)
            client_writer.write(response_wire)
            await client_writer.drain()
            response = self._build_result(parsed_response, response_wire)

        request_wire_bytes = header_wire_bytes + bytes(remaining_buffer)
        return ForwardedRequest(
            parsed_request=parsed_headers,
            request_wire_bytes=request_wire_bytes,
            response=response,
            # A bodyless request (e.g. Content-Length: 0) is already
            # complete at the header boundary, so an early final
            # response leaves nothing unread on the connection.
            request_body_consumed=parsed_headers.is_complete,
        )

    async def _relay_until_continue_or_final(
        self,
        *,
        parser: AsyncMultiResponseParser,
        upstream_reader: asyncio.StreamReader,
        client_writer: ClientWriter,
        request_method: str | None,
        upstream_label: str,
        client_label: str,
    ) -> tuple[ParsedResponse | None, bytes]:
        while True:
            parsed, wire_bytes = await parser.next_response(
                upstream_reader,
                request_method,
            )
            if parsed is None:
                return None, wire_bytes

            status = parsed.status_code or 0
            if status == 101 or status >= 200:
                return parsed, wire_bytes

            if self._wire_log is not None:
                self._wire_log(f"{upstream_label} --> lstub", wire_bytes)
                self._wire_log(f"lstub --> {client_label}", wire_bytes)
            client_writer.write(wire_bytes)
            await client_writer.drain()

            if status == 100:
                return parsed, wire_bytes

    async def _apply_transformation_and_relay(
        self,
        *,
        parsed: ParsedResponse,
        wire_bytes: bytes,
        client_writer: ClientWriter,
        client_label: str,
    ) -> ForwardResult:
        headers = headers_to_message(parsed.headers)
        body_bytes = self._maybe_decompress(headers, parsed.body)
        reason = decode_status_text(parsed.status_text)

        context = TransformContext(
            status=parsed.status_code or 0,
            reason=reason,
            headers=headers,
            body=body_bytes,
            wire_raw_bytes=wire_bytes,
        )

        assert self._response_transformer is not None
        result = await maybe_await(self._response_transformer(context))

        if result.delay_before > 0:
            await asyncio.sleep(result.delay_before)

        wire_bytes_to_send = self._build_wire_bytes_for_result(
            result=result,
            parsed=parsed,
            original_wire_bytes=wire_bytes,
        )

        if result.drop_after is not None:
            partial = wire_bytes_to_send[: result.drop_after]
            if self._wire_log is not None:
                self._wire_log(f"lstub --> {client_label}", partial)
            client_writer.write(partial)
            await client_writer.drain()
            client_writer.close()
            try:
                await client_writer.wait_closed()
            except Exception:
                LOG.debug(
                    "Failed to close client writer",
                    exc_info=True,
                )
        else:
            if self._wire_log is not None:
                self._wire_log(f"lstub --> {client_label}", wire_bytes_to_send)
            client_writer.write(wire_bytes_to_send)
            await client_writer.drain()

        return self._build_result(
            parsed,
            wire_bytes,
            relayed=self._relayed_metadata(result, parsed, headers),
        )

    def _relayed_metadata(
        self,
        result: TransformResult,
        parsed: ParsedResponse,
        upstream_headers: Message,
    ) -> RelayedResponse:
        """Describe the response actually sent after transformation."""
        if result.override_response is not None:
            # An override is serialized with an HTTP/1.1 status line
            # and Content-Length framing.
            return RelayedResponse(
                status=result.override_response.status,
                headers=result.override_response.headers,
                http_version="1.1",
                is_eof_delimited=False,
            )
        if result.body is not None:
            # A rewritten body is reframed with Content-Length while
            # keeping the upstream status line and headers.
            return RelayedResponse(
                status=parsed.status_code or 0,
                headers=upstream_headers,
                http_version=parsed.http_version,
                is_eof_delimited=False,
            )
        return RelayedResponse(
            status=parsed.status_code or 0,
            headers=upstream_headers,
            http_version=parsed.http_version,
            is_eof_delimited=parsed.is_eof_delimited,
        )

    def _maybe_decompress(self, headers: Message, body: bytes) -> bytes:
        if not self._decompress_body:
            return body
        content_encoding = headers.get("Content-Encoding", "").lower()
        if "gzip" not in content_encoding:
            return body
        try:
            return gzip.decompress(body)
        except Exception:
            LOG.debug(
                "Failed to decompress gzip response body",
                exc_info=True,
            )
            return body

    def _rebuild_response_wire_bytes(
        self,
        parsed: ParsedResponse,
        new_body: bytes,
    ) -> bytes:
        version = parsed.http_version or "1.1"
        status_code = parsed.status_code or 200
        status_text = decode_status_text(parsed.status_text, "OK")
        head = bytearray(
            f"HTTP/{version} {status_code} {status_text}\r\n".encode("ascii")
        )

        skip_headers = {
            b"content-length",
            b"transfer-encoding",
            b"content-encoding",
        }
        for name, value in parsed.headers:
            if name.lower() in skip_headers:
                continue
            head.extend(name + b": " + value + b"\r\n")

        head.extend(f"Content-Length: {len(new_body)}\r\n\r\n".encode("ascii"))
        return bytes(head) + new_body

    def _build_override_wire_bytes(self, response: HTTPResponse) -> bytes:
        reason = status_phrase(response.status, "UNKNOWN")

        head = bytearray(
            f"HTTP/1.1 {response.status} {reason}\r\n".encode("ascii")
        )

        if isinstance(response.body, str):
            body = response.body.encode("utf-8")
        else:
            body = response.body

        items = list(response.headers.items())
        if "Content-Length" not in response.headers:
            items.append(("Content-Length", str(len(body))))

        for name, value in items:
            head.extend(serialize_header_line(name, value))

        head.extend(b"\r\n")
        return bytes(head) + body

    def _build_wire_bytes_for_result(
        self,
        *,
        result: TransformResult,
        parsed: ParsedResponse,
        original_wire_bytes: bytes,
    ) -> bytes:
        if result.override_response is not None:
            return self._build_override_wire_bytes(result.override_response)
        if result.body is not None:
            return self._rebuild_response_wire_bytes(parsed, result.body)
        return original_wire_bytes

    def _build_result(
        self,
        parsed: ParsedResponse,
        wire_bytes: bytes,
        relayed: RelayedResponse | None = None,
    ) -> ForwardResult:
        """Build ForwardResult from parsed response.

        Args:
            parsed: Parsed response from upstream.
            wire_bytes: Raw wire bytes of the response.
            relayed: Metadata of the response sent to the client, when
                it differs from the upstream response.  Defaults to the
                upstream response relayed verbatim.

        Returns:
            ForwardResult for recording.
        """
        headers = headers_to_message(parsed.headers)
        body = self._maybe_decompress(headers, parsed.body)
        if relayed is None:
            relayed = RelayedResponse(
                status=parsed.status_code or 0,
                headers=headers,
                http_version=parsed.http_version,
                is_eof_delimited=parsed.is_eof_delimited,
            )
        return ForwardResult(
            status=parsed.status_code or 0,
            reason=decode_status_text(parsed.status_text),
            headers=headers,
            body=body,
            wire_bytes=wire_bytes,
            relayed=relayed,
            is_eof_delimited=parsed.is_eof_delimited,
        )
