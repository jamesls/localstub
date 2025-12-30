"""Shared HTTP forwarding logic for proxy modes.

This module provides the Forwarder class which handles forwarding HTTP
requests to upstream servers and relaying responses back to clients,
preserving exact wire bytes including Transfer-Encoding.
"""

from __future__ import annotations

import asyncio
import gzip
import inspect
import logging
import ssl
from enum import Enum
from dataclasses import dataclass
from email.message import Message
from http import HTTPStatus
from typing import Awaitable, Callable, Protocol

from localstub.http.request import AsyncRequestParser, ParsedRequest
from localstub.http.response import AsyncMultiResponseParser, ParsedResponse
from localstub.http.utils import headers_to_message


LOG = logging.getLogger(__name__)

WireLog = Callable[[str, bytes], None]


class OverrideResponse(Protocol):
    """Protocol for response objects that can override an upstream response."""

    status: int
    headers: dict[str, str]
    body: bytes | str | None


@dataclass
class UpstreamResponse:
    """Upstream response context provided to transformers."""

    status: int
    reason: str | None
    headers: Message
    body: bytes
    wire_raw_bytes: bytes


@dataclass
class TransformResult:
    """Result of transforming an upstream response."""

    body: bytes | None = None
    override_response: OverrideResponse | None = None
    delay_before: float = 0.0
    drop_after: int | None = None


ResponseTransformer = Callable[
    [UpstreamResponse],
    TransformResult | Awaitable[TransformResult],
]


@dataclass
class ForwardResult:
    """Result of forwarding a request to upstream."""

    status: int
    reason: str | None
    headers: Message
    body: bytes
    wire_bytes: bytes


class ForwardError(Enum):
    """Forwarding error category."""

    REQUEST_PARSE_FAILED = "request_parse_failed"
    RESPONSE_PARSE_FAILED = "response_parse_failed"


@dataclass
class ForwardedRequest:
    """Result of forwarding a request that originated from a client."""

    parsed_request: ParsedRequest
    request_wire_bytes: bytes
    response: ForwardResult | None
    error: ForwardError | None = None


class Forwarder:
    """Forwards HTTP requests to upstream servers and relays responses.

    This class provides raw socket-based forwarding that preserves exact
    wire bytes, including Transfer-Encoding headers and chunked framing.
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
        client_writer: asyncio.StreamWriter,
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
            await upstream_writer.wait_closed()

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
        ssl_ctx: ssl.SSLContext | None = None
        server_hostname: str | None = None
        if use_tls:
            ssl_ctx = ssl.create_default_context()
            if not self._verify_upstream:
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
            server_hostname = host
        try:
            return await asyncio.open_connection(
                host, port, ssl=ssl_ctx, server_hostname=server_hostname
            )
        except Exception as e:
            LOG.warning(
                "Failed to connect to upstream %s:%d: %s", host, port, e
            )
            return None

    async def read_and_relay_responses(
        self,
        upstream_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        request_method: str | None,
        *,
        upstream_id: str | None = None,
        client_id: str | None = None,
        multi_parser: AsyncMultiResponseParser | None = None,
    ) -> ForwardResult | None:
        """Read responses from upstream and relay to client.

        Handles 1xx informational responses by relaying them and continuing
        to read until a final (2xx+) response is received.

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

            # Check if this is a final response (not 1xx informational)
            if status >= 200:
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
        client_writer: asyncio.StreamWriter,
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

        interim, interim_wire = await resp_parser.next_response(
            upstream_reader, request_method
        )
        if interim is None:
            return ForwardedRequest(
                parsed_request=parsed_headers,
                request_wire_bytes=(
                    header_wire_bytes + bytes(remaining_buffer)
                ),
                response=None,
                error=ForwardError.RESPONSE_PARSE_FAILED,
            )

        status = interim.status_code or 0
        if status < 200:
            if self._wire_log is not None:
                self._wire_log(f"{upstream_label} --> lstub", interim_wire)
                self._wire_log(f"lstub --> {client_label}", interim_wire)
            client_writer.write(interim_wire)
            await client_writer.drain()

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
                error=(
                    None
                    if response is not None
                    else ForwardError.RESPONSE_PARSE_FAILED
                ),
            )

        response: ForwardResult | None
        if self._response_transformer is not None:
            response = await self._apply_transformation_and_relay(
                parsed=interim,
                wire_bytes=interim_wire,
                client_writer=client_writer,
                client_label=client_label,
            )
        else:
            if self._wire_log is not None:
                self._wire_log(f"{upstream_label} --> lstub", interim_wire)
                self._wire_log(f"lstub --> {client_label}", interim_wire)
            client_writer.write(interim_wire)
            await client_writer.drain()
            response = self._build_result(interim, interim_wire)

        request_wire_bytes = header_wire_bytes + bytes(remaining_buffer)
        return ForwardedRequest(
            parsed_request=parsed_headers,
            request_wire_bytes=request_wire_bytes,
            response=response,
        )

    async def _apply_transformation_and_relay(
        self,
        *,
        parsed: ParsedResponse,
        wire_bytes: bytes,
        client_writer: asyncio.StreamWriter,
        client_label: str,
    ) -> ForwardResult:
        headers = headers_to_message(parsed.headers)
        body_bytes = self._maybe_decompress(headers, parsed.body)
        reason = (
            parsed.status_text.decode("ascii", errors="replace")
            if parsed.status_text
            else None
        )

        upstream = UpstreamResponse(
            status=parsed.status_code or 0,
            reason=reason,
            headers=headers,
            body=body_bytes,
            wire_raw_bytes=wire_bytes,
        )

        assert self._response_transformer is not None
        result = self._response_transformer(upstream)
        if inspect.isawaitable(result):
            result = await result

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
                pass
        else:
            if self._wire_log is not None:
                self._wire_log(f"lstub --> {client_label}", wire_bytes_to_send)
            client_writer.write(wire_bytes_to_send)
            await client_writer.drain()

        return self._build_result(parsed, wire_bytes)

    def _maybe_decompress(self, headers: Message, body: bytes) -> bytes:
        if not self._decompress_body:
            return body
        content_encoding = headers.get("Content-Encoding", "").lower()
        if "gzip" not in content_encoding:
            return body
        try:
            return gzip.decompress(body)
        except Exception:
            return body

    def _rebuild_response_wire_bytes(
        self,
        parsed: ParsedResponse,
        new_body: bytes,
    ) -> bytes:
        version = parsed.http_version or "1.1"
        status_code = parsed.status_code or 200
        status_text = (
            parsed.status_text.decode("ascii", errors="replace")
            if parsed.status_text
            else "OK"
        )
        lines = [f"HTTP/{version} {status_code} {status_text}"]

        skip_headers = {
            b"content-length",
            b"transfer-encoding",
            b"content-encoding",
        }
        for name, value in parsed.headers:
            if name.lower() in skip_headers:
                continue
            header_name = name.decode("ascii", errors="replace")
            header_value = value.decode("ascii", errors="replace")
            lines.append(f"{header_name}: {header_value}")

        lines.append(f"Content-Length: {len(new_body)}")
        lines.append("")
        header_bytes = "\r\n".join(lines).encode("ascii") + b"\r\n"
        return header_bytes + new_body

    def _build_override_wire_bytes(self, response: OverrideResponse) -> bytes:
        try:
            reason = HTTPStatus(response.status).phrase
        except ValueError:
            reason = "UNKNOWN"

        lines = [f"HTTP/1.1 {response.status} {reason}"]

        if isinstance(response.body, str):
            body = response.body.encode("utf-8")
        elif response.body is None:
            body = b""
        else:
            body = response.body

        headers = dict(response.headers)
        if "Content-Length" not in headers and "content-length" not in headers:
            headers["Content-Length"] = str(len(body))

        for name, value in headers.items():
            lines.append(f"{name}: {value}")

        lines.append("")
        header_bytes = "\r\n".join(lines).encode("ascii") + b"\r\n"
        return header_bytes + body

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
    ) -> ForwardResult:
        """Build ForwardResult from parsed response.

        Args:
            parsed: Parsed response from upstream.
            wire_bytes: Raw wire bytes of the response.

        Returns:
            ForwardResult for recording.
        """
        headers = headers_to_message(parsed.headers)
        body = self._maybe_decompress(headers, parsed.body)
        return ForwardResult(
            status=parsed.status_code or 0,
            reason=(
                parsed.status_text.decode("ascii", errors="replace")
                if parsed.status_text
                else None
            ),
            headers=headers,
            body=body,
            wire_bytes=wire_bytes,
        )
