from __future__ import annotations

import inspect
import ssl
import asyncio
import logging
from asyncio import transports
import gzip
from dataclasses import dataclass
from email.message import Message
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, cast

if TYPE_CHECKING:
    from localstub.server import FaultStep

from localstub.http.response import (
    AsyncMultiResponseParser,
    AsyncResponseParser,
    ParsedResponse,
)
from localstub.http.request import (
    AsyncRequestParser,
    HTTPRequest,
    ParsedRequest,
)
from localstub.http.utils import headers_to_message
from localstub.ca import TLSProxyCA
from localstub.server import AsyncHTTPTestServer, HTTPResponse


LOG = logging.getLogger(__name__)


def _wire_log(direction: str, data: bytes) -> None:
    """Log wire-level data with direction indicator."""
    text = data.decode("utf-8", errors="replace").rstrip("\r\n")
    # Truncate long messages for readability
    if len(text) > 200:
        text = text[:200] + "..."
    LOG.debug("[%s] %s", direction, text)


def _close_log(direction: str, reason: str) -> None:
    """Log connection closure with direction and reason."""
    LOG.debug("[%s CLOSED] %s", direction, reason)


class TLSStreamReaderProtocol(asyncio.StreamReaderProtocol):
    """Protocol variant that suppresses SSL EOF warning.

    Feed EOF to the attached reader and return ``False`` so asyncio's
    SSL layer does not warn about half‑closed behavior under TLS.
    """

    def __init__(self, stream_reader: asyncio.StreamReader) -> None:
        super().__init__(stream_reader)
        self._reader_ref = stream_reader

    def eof_received(self) -> bool:
        reader = self._reader_ref
        if reader is not None:
            reader.feed_eof()
        return False


class AsyncTLSInterceptProxy:
    """Minimal TLS intercept proxy that routes CONNECT traffic to localstub."""

    def __init__(
        self,
        *,
        listen_host: str = "127.0.0.1",
        listen_port: int = 0,
        server: Optional[AsyncHTTPTestServer] = None,
        ca: Optional[TLSProxyCA] = None,
        max_read: int = 8192,
        default_mode: str = "intercept",
        verify_upstream: bool = True,
        upstream_tls: bool = True,
        response_transformer: "ResponseTransformer | None" = None,
    ) -> None:
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._server = server
        self._ca = ca or TLSProxyCA()
        self._max_read = max_read
        self._default_mode = default_mode
        self._verify_upstream = verify_upstream
        self._upstream_tls = upstream_tls
        self._response_transformer = response_transformer

        self._listener: asyncio.base_events.Server | None = None
        self._host: str | None = None
        self._port: int | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()

        # Recording queues for forwarded traffic
        self._recorded_requests: asyncio.Queue[HTTPRequest] = asyncio.Queue()
        self._recorded_responses: asyncio.Queue["RecordedResponse"] = (
            asyncio.Queue()
        )

    @property
    def address(self) -> tuple[str, int]:
        if self._host is None or self._port is None:
            raise RuntimeError("Proxy not started yet")
        return (self._host, self._port)

    @property
    def endpoint_url(self) -> str:
        if self._host is None or self._port is None:
            raise RuntimeError("Proxy not started yet")
        return f'http://{self._host}:{self._port}'

    @property
    def ca(self) -> TLSProxyCA:
        return self._ca

    async def next_request(self, timeout: float | None = None) -> HTTPRequest:
        if timeout is None:
            return await self._recorded_requests.get()
        return await asyncio.wait_for(
            self._recorded_requests.get(), timeout=timeout
        )

    async def next_response(
        self, timeout: float | None = None
    ) -> "RecordedResponse":
        if timeout is None:
            return await self._recorded_responses.get()
        return await asyncio.wait_for(
            self._recorded_responses.get(), timeout=timeout
        )

    async def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = await asyncio.start_server(
            self._client_connected,
            self._listen_host,
            self._listen_port,
        )
        assert self._listener.sockets
        sockname = self._listener.sockets[0].getsockname()
        self._host, self._port = sockname[0], sockname[1]

    async def aclose(self) -> None:
        if self._listener is None:
            return
        self._listener.close()
        await self._listener.wait_closed()
        # Wait for in-flight client handlers to finish; cancel any that linger.
        pending = [t for t in self._client_tasks if not t.done()]
        if pending:
            done, pending = await asyncio.wait(pending, timeout=1.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
        self._client_tasks.clear()
        self._listener = None

    async def __aenter__(self) -> "AsyncTLSInterceptProxy":
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        # Extract client port for logging
        peername = writer.get_extra_info("peername")
        client_port = peername[1] if peername else 0
        client_id = f"client:{client_port}"

        try:
            connect_host, connect_port = await self._parse_connect(
                reader, client_id
            )
            if connect_host is None or connect_port is None:
                await self._send_and_close(
                    writer, b"HTTP/1.1 400 Bad Request", client_id
                )
                return

            response_bytes = b"HTTP/1.1 200 Connection Established\r\n\r\n"
            _wire_log(f"lstub --> {client_id}", response_bytes)
            writer.write(response_bytes)
            await writer.drain()

            try:
                tls_reader, tls_writer = await self._upgrade_to_tls(
                    writer, connect_host
                )
            except (ConnectionResetError, ssl.SSLError) as exc:
                # Many real clients immediately drop the connection if they
                # don't trust our ephemeral CA. Treat this as a normal
                # condition and avoid a noisy stack trace; provide a helpful
                # hint instead.
                LOG.warning(
                    "TLS handshake from client failed for %s:%s; "
                    "client likely rejected the proxy CA (%s): %s",
                    connect_host,
                    connect_port,
                    self._ca.ca_pem_path(),
                    exc,
                )
                try:
                    _close_log(
                        f"lstub --> {client_id}", "TLS handshake failed"
                    )
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
                return

            if self._default_mode == "forward":
                await self._forward(
                    connect_host,
                    connect_port,
                    tls_reader,
                    tls_writer,
                    client_id,
                )
                return

            if self._server is None:
                await self._send_and_close(
                    tls_writer, b"HTTP/1.1 502 Bad Gateway", client_id
                )
                return

            await self._server.handle_http_connection(tls_reader, tls_writer)
        except asyncio.CancelledError:
            try:
                _close_log(f"lstub --> {client_id}", "task cancelled")
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            raise
        except Exception:
            # Any other exception is unexpected; keep the traceback to aid
            # debugging.
            LOG.exception("TLS proxy error")
            try:
                _close_log(f"lstub --> {client_id}", "unexpected error")
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _client_connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._client_tasks.add(task)
        task.add_done_callback(self._client_tasks.discard)

    async def _parse_connect(
        self, reader: asyncio.StreamReader, client_id: str
    ) -> tuple[str | None, int | None]:
        line = await reader.readline()
        if not line:
            return None, None
        _wire_log(f"lstub <-- {client_id}", line)
        try:
            req_line = line.decode("ascii", errors="replace").strip()
            parts = req_line.split(" ")
            if len(parts) < 3:
                return None, None
            method, target, _version = parts[0], parts[1], parts[2]
            if method.upper() != "CONNECT":
                return None, None
            host, port_str = self._split_connect_target(target)
            port = int(port_str)
        except Exception:
            return None, None

        # Consume remaining headers up to blank line
        while True:
            header_line = await reader.readline()
            if not header_line:
                break
            _wire_log(f"lstub <-- {client_id}", header_line)
            if header_line in (b"\r\n", b"\n"):
                break
        return host, port

    def _split_connect_target(self, target: str) -> tuple[str, str]:
        if target.startswith("["):
            closing = target.rfind("]")
            if closing == -1:
                raise ValueError("Invalid IPv6 target")
            host = target[1:closing]
            remainder = target[closing + 1 :]
            if remainder.startswith(":"):
                port_str = remainder[1:] or "443"
            elif remainder == "":
                port_str = "443"
            else:
                raise ValueError("Invalid IPv6 target")
            return host, port_str

        if ":" in target:
            host, port_str = target.rsplit(":", 1)
        else:
            host, port_str = target, "443"
        return host, port_str

    async def _upgrade_to_tls(
        self,
        writer: asyncio.StreamWriter,
        host: str,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        loop = asyncio.get_running_loop()
        ssl_context = self._ca.issue_context(host)

        transport = writer.transport
        if transport is None:
            raise RuntimeError(
                "Stream transport unavailable during TLS upgrade"
            )

        tls_reader = asyncio.StreamReader()
        tls_protocol = TLSStreamReaderProtocol(tls_reader)

        tls_transport = await loop.start_tls(
            transport,
            tls_protocol,
            ssl_context,
            server_side=True,
            ssl_handshake_timeout=10.0,
        )

        tls_writer = asyncio.StreamWriter(
            cast(transports.WriteTransport, tls_transport),
            tls_protocol,
            tls_reader,
            loop,
        )
        return tls_reader, tls_writer

    async def _read_upstream_response(
        self,
        reader: asyncio.StreamReader,
        request_method: str | None = None,
    ) -> RecordedResponse | None:
        parser = AsyncResponseParser(max_read=self._max_read)
        parsed, wire_bytes = await parser.parse(reader, request_method)

        if parsed is None:
            return None

        headers = headers_to_message(parsed.headers)
        body_bytes = self._maybe_decompress(headers, parsed.body)
        body_text = body_bytes.decode("utf-8", errors="replace")

        return RecordedResponse(
            status=parsed.status_code or 0,
            reason=(
                parsed.status_text.decode("ascii", errors="replace")
                if parsed.status_text
                else None
            ),
            headers=headers,
            body=body_text,
            wire_raw_bytes=wire_bytes,
        )

    def _maybe_decompress(self, headers: Message, body: bytes) -> bytes:
        content_encoding = headers.get("Content-Encoding", "").lower()
        if "gzip" in content_encoding:
            try:
                return gzip.decompress(body)
            except Exception:
                return body
        return body

    def _rebuild_response_wire_bytes(
        self,
        parsed: ParsedResponse,
        new_body: bytes,
    ) -> bytes:
        """Rebuild HTTP response with new body, using identity encoding.

        Strips Transfer-Encoding and Content-Encoding headers since the
        new body is sent in plain format with Content-Length.
        """
        version = parsed.http_version or "1.1"
        status_code = parsed.status_code or 200
        status_text = (
            parsed.status_text.decode("ascii", errors="replace")
            if parsed.status_text
            else "OK"
        )
        lines = [f"HTTP/{version} {status_code} {status_text}"]

        # Filter out Content-Length, Transfer-Encoding, Content-Encoding
        # Then add new Content-Length
        skip_headers = {
            b"content-length",
            b"transfer-encoding",
            b"content-encoding",
        }
        for name, value in parsed.headers:
            if name.lower() not in skip_headers:
                lines.append(
                    f"{name.decode('ascii', errors='replace')}: "
                    f"{value.decode('ascii', errors='replace')}"
                )

        lines.append(f"Content-Length: {len(new_body)}")
        lines.append("")  # Blank line before body

        header_bytes = "\r\n".join(lines).encode("ascii") + b"\r\n"
        return header_bytes + new_body

    def _build_override_wire_bytes(self, response: HTTPResponse) -> bytes:
        """Build wire bytes from an HTTPResponse for full response override."""
        try:
            from http import HTTPStatus

            reason = HTTPStatus(response.status).phrase
        except ValueError:
            reason = "UNKNOWN"

        lines = [f"HTTP/1.1 {response.status} {reason}"]

        body: bytes
        if isinstance(response.body, str):
            body = response.body.encode("utf-8")
        elif response.body is None:
            body = b""
        else:
            body = response.body

        # Add headers, ensuring Content-Length
        headers = dict(response.headers) if response.headers else {}
        if "Content-Length" not in headers and "content-length" not in headers:
            headers["Content-Length"] = str(len(body))

        for name, value in headers.items():
            lines.append(f"{name}: {value}")

        lines.append("")
        header_bytes = "\r\n".join(lines).encode("ascii") + b"\r\n"
        return header_bytes + body

    def _build_wire_bytes_for_result(
        self,
        result: "TransformResult",
        parsed: ParsedResponse,
        original_wire_bytes: bytes,
    ) -> bytes:
        """Build wire bytes based on transformer result."""
        if result.override_response is not None:
            return self._build_override_wire_bytes(result.override_response)
        if result.body is not None:
            return self._rebuild_response_wire_bytes(parsed, result.body)
        return original_wire_bytes

    async def _apply_transformation_and_relay(
        self,
        parsed: ParsedResponse,
        wire_bytes: bytes,
        client_writer: asyncio.StreamWriter,
        client_id: str,
    ) -> RecordedResponse:
        """Apply response transformation and relay to client.

        Returns the original (untransformed) response for recording.
        """
        status = parsed.status_code or 0
        headers = headers_to_message(parsed.headers)
        body_bytes = self._maybe_decompress(headers, parsed.body)
        reason = (
            parsed.status_text.decode("ascii", errors="replace")
            if parsed.status_text
            else None
        )

        upstream = UpstreamResponse(
            status=status,
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
            result, parsed, wire_bytes
        )

        if result.drop_after is not None:
            partial = wire_bytes_to_send[: result.drop_after]
            _wire_log(f"lstub --> {client_id}", partial)
            client_writer.write(partial)
            await client_writer.drain()
            reason = f"drop_after={result.drop_after}"
            _close_log(f"lstub --> {client_id}", reason)
            client_writer.close()
            try:
                await client_writer.wait_closed()
            except Exception:
                pass
        else:
            _wire_log(f"lstub --> {client_id}", wire_bytes_to_send)
            client_writer.write(wire_bytes_to_send)
            await client_writer.drain()

        body_text = body_bytes.decode("utf-8", errors="replace")
        return RecordedResponse(
            status=status,
            reason=reason,
            headers=headers,
            body=body_text,
            wire_raw_bytes=wire_bytes,
        )

    async def _read_and_relay_responses(
        self,
        upstream_reader: asyncio.StreamReader,
        upstream_id: str,
        client_writer: asyncio.StreamWriter,
        client_id: str,
        request_method: str | None,
        *,
        multi_parser: AsyncMultiResponseParser | None = None,
    ) -> RecordedResponse | None:
        """Read responses from upstream, relaying each to client.

        Handles 1xx informational responses by relaying them and continuing
        to read until a final (2xx+) response is received.

        Uses a single parser instance to handle cases where multiple responses
        (e.g., 100 Continue + 200 OK) arrive in a single buffer read. If
        a parser is provided via ``multi_parser``, it will be reused so any
        already-buffered upstream bytes are not lost.

        If a response_transformer is configured, it will be applied to
        final responses (status >= 200) before relaying to the client.

        Returns the final response, or None if parsing failed.
        """
        if multi_parser is None:
            multi_parser = AsyncMultiResponseParser(max_read=self._max_read)

        while True:
            parsed, wire_bytes = await multi_parser.next_response(
                upstream_reader, request_method
            )
            if parsed is None:
                return None

            _wire_log(f"{upstream_id} --> lstub", wire_bytes)

            status = parsed.status_code or 0

            # Only transform final responses (not 1xx informational)
            if self._response_transformer is not None and status >= 200:
                return await self._apply_transformation_and_relay(
                    parsed, wire_bytes, client_writer, client_id
                )

            # No transformation - relay original wire bytes
            _wire_log(f"lstub --> {client_id}", wire_bytes)
            client_writer.write(wire_bytes)
            await client_writer.drain()

            # Check if this is a final response (not 1xx informational)
            if status >= 200:
                headers = headers_to_message(parsed.headers)
                body_bytes = self._maybe_decompress(headers, parsed.body)
                body_text = body_bytes.decode("utf-8", errors="replace")
                return RecordedResponse(
                    status=status,
                    reason=(
                        parsed.status_text.decode("ascii", errors="replace")
                        if parsed.status_text
                        else None
                    ),
                    headers=headers,
                    body=body_text,
                    wire_raw_bytes=wire_bytes,
                )

            # For 1xx responses, log and continue reading for final response
            LOG.debug(
                "Received 1xx informational response (%d), "
                "waiting for final response",
                status,
            )

    async def _forward(
        self,
        host: str,
        port: int,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        client_id: str,
    ) -> None:
        upstream_id = f"{host}:{port}"

        # Parse headers first to check for Expect: 100-continue.
        parser = AsyncRequestParser(max_read=self._max_read)
        parsed, header_wire, remaining = await parser.parse_headers(
            client_reader
        )
        if parsed is None:
            await self._send_and_close(
                client_writer, b"HTTP/1.1 400 Bad Request", client_id
            )
            return

        _wire_log(f"lstub <-- {client_id}", header_wire)

        # Check for Expect: 100-continue with client actually waiting.
        expect_hdr = dict((k.lower(), v) for k, v in parsed.headers).get(
            b"expect", b""
        )
        client_waiting = (
            b"100-continue" in expect_hdr.lower() and len(remaining) == 0
        )

        # Connect to upstream.
        upstream = await self._connect_upstream(host, port)
        if upstream is None:
            # Connection failed - record what we have without blocking on the
            # body (e.g. Expect: 100-continue clients may not send it yet).
            wire_bytes = header_wire + bytes(remaining)
            await self._record_request(parsed, wire_bytes, client_writer)
            await self._send_and_close(
                client_writer, b"HTTP/1.1 502 Bad Gateway", client_id
            )
            return
        upstream_reader, upstream_writer = upstream

        response_parser: AsyncMultiResponseParser | None = None

        # Forward request based on whether client is waiting for 100-continue.
        if client_waiting:
            response_parser = AsyncMultiResponseParser(max_read=self._max_read)
            done = await self._forward_with_100_continue(
                parser,
                parsed,
                header_wire,
                remaining,
                client_reader,
                client_writer,
                client_id,
                upstream_reader,
                upstream_writer,
                upstream_id,
                response_parser,
            )
            if done:
                return
        else:
            ok = await self._forward_normal(
                parser,
                header_wire,
                remaining,
                client_reader,
                client_writer,
                client_id,
                upstream_writer,
                upstream_id,
            )
            if not ok:
                return

        # Read and relay responses.
        final_response = await self._read_and_relay_responses(
            upstream_reader,
            upstream_id,
            client_writer,
            client_id,
            parsed.method,
            multi_parser=response_parser,
        )
        if final_response is None:
            await self._send_and_close(
                client_writer, b"HTTP/1.1 502 Bad Gateway", client_id
            )
            _close_log(f"{upstream_id} --> lstub", "failed to parse response")
            upstream_writer.close()
            return

        await self._recorded_responses.put(final_response)
        _close_log(f"{upstream_id} --> lstub", "response received")
        upstream_writer.close()
        _close_log(f"lstub --> {client_id}", "response relayed")
        client_writer.close()

    async def _connect_upstream(
        self, host: str, port: int
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
        """Connect to upstream server. Returns None on failure."""
        ssl_param: ssl.SSLContext | None = None
        server_hostname: str | None = None
        if self._upstream_tls:
            ctx = ssl.create_default_context()
            if not self._verify_upstream:
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
            ssl_param = ctx
            server_hostname = host
        try:
            return await asyncio.open_connection(
                host, port, ssl=ssl_param, server_hostname=server_hostname
            )
        except Exception:
            return None

    async def _forward_with_100_continue(
        self,
        parser: AsyncRequestParser,
        parsed: ParsedRequest,
        header_wire: bytes,
        remaining: bytearray,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        client_id: str,
        upstream_reader: asyncio.StreamReader,
        upstream_writer: asyncio.StreamWriter,
        upstream_id: str,
        resp_parser: AsyncMultiResponseParser,
    ) -> bool:
        """Handle forwarding when client waits for 100-continue.

        Returns True if request is complete (caller should return),
        False to continue with normal response handling.
        """
        # Forward headers first.
        _wire_log(f"{upstream_id} <-- lstub", header_wire)
        upstream_writer.write(header_wire)
        await upstream_writer.drain()

        # Get upstream's response (100 Continue or error).
        interim, interim_wire = await resp_parser.next_response(
            upstream_reader, parsed.method
        )
        if interim is None:
            await self._send_and_close(
                client_writer, b"HTTP/1.1 502 Bad Gateway", client_id
            )
            _close_log(f"{upstream_id} --> lstub", "failed to parse response")
            upstream_writer.close()
            return True

        # Relay interim response to client.
        _wire_log(f"{upstream_id} --> lstub", interim_wire)
        _wire_log(f"lstub --> {client_id}", interim_wire)
        client_writer.write(interim_wire)
        await client_writer.drain()

        if (interim.status_code or 0) >= 200:
            # Final response (e.g., 417). Record and close.
            await self._record_request(parsed, header_wire, client_writer)
            await self._record_response_and_close(
                interim,
                interim_wire,
                upstream_writer,
                client_writer,
                upstream_id,
                client_id,
            )
            return True

        # Got 1xx - now read body from client.
        final_parsed, full_wire = await parser.continue_parse_body(
            client_reader, remaining
        )
        if final_parsed is None:
            await self._send_and_close(
                client_writer, b"HTTP/1.1 400 Bad Request", client_id
            )
            upstream_writer.close()
            return True

        body_wire = full_wire[len(header_wire) :]
        if body_wire:
            _wire_log(f"lstub <-- {client_id}", body_wire)
            _wire_log(f"{upstream_id} <-- lstub", body_wire)
            upstream_writer.write(body_wire)
            await upstream_writer.drain()

        await self._record_request(final_parsed, full_wire, client_writer)
        return False

    async def _forward_normal(
        self,
        parser: AsyncRequestParser,
        header_wire: bytes,
        remaining: bytearray,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        client_id: str,
        upstream_writer: asyncio.StreamWriter,
        upstream_id: str,
    ) -> bool:
        """Normal forwarding: read full body, forward complete request.

        Returns True on success, False on failure.
        """
        final_parsed, full_wire = await parser.continue_parse_body(
            client_reader, remaining
        )
        if final_parsed is None:
            await self._send_and_close(
                client_writer, b"HTTP/1.1 400 Bad Request", client_id
            )
            upstream_writer.close()
            return False

        body_wire = full_wire[len(header_wire) :]
        if body_wire:
            _wire_log(f"lstub <-- {client_id}", body_wire)

        await self._record_request(final_parsed, full_wire, client_writer)
        _wire_log(f"{upstream_id} <-- lstub", full_wire)
        upstream_writer.write(full_wire)
        await upstream_writer.drain()
        return True

    async def _record_request(
        self,
        parsed: ParsedRequest,
        wire_bytes: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Build and record an HTTPRequest."""
        headers = headers_to_message(parsed.headers)
        body = (
            parsed.body.decode("utf-8", errors="replace")
            if parsed.body
            else None
        )
        peer = writer.get_extra_info("peername")
        client = (peer[0], peer[1]) if isinstance(peer, tuple) else None
        path = (
            parsed.url.decode("ascii", errors="replace")
            if parsed.url
            else None
        )
        request = HTTPRequest(
            method=parsed.method,
            path=path,
            http_version=parsed.http_version,
            headers=headers,
            body=body,
            wire_raw_bytes=wire_bytes,
            client=client,
        )
        await self._recorded_requests.put(request)

    async def _record_response_and_close(
        self,
        parsed: ParsedResponse,
        wire_bytes: bytes,
        upstream_writer: asyncio.StreamWriter,
        client_writer: asyncio.StreamWriter,
        upstream_id: str,
        client_id: str,
    ) -> None:
        """Record response and close connections."""
        headers = headers_to_message(parsed.headers)
        body_bytes = self._maybe_decompress(headers, parsed.body)
        reason = (
            parsed.status_text.decode("ascii", errors="replace")
            if parsed.status_text
            else None
        )
        response = RecordedResponse(
            status=parsed.status_code or 0,
            reason=reason,
            headers=headers,
            body=body_bytes.decode("utf-8", errors="replace"),
            wire_raw_bytes=wire_bytes,
        )
        await self._recorded_responses.put(response)
        _close_log(f"{upstream_id} --> lstub", "response received")
        upstream_writer.close()
        _close_log(f"lstub --> {client_id}", "response relayed")
        client_writer.close()

    async def _send_and_close(
        self,
        writer: asyncio.StreamWriter,
        status_line: bytes,
        client_id: str,
    ) -> None:
        try:
            response_bytes = status_line + b"\r\n\r\n"
            _wire_log(f"lstub --> {client_id}", response_bytes)
            writer.write(response_bytes)
            await writer.drain()
        finally:
            reason = status_line.decode("ascii", errors="replace")
            _close_log(f"lstub --> {client_id}", f"sent {reason}")
            writer.close()
            if hasattr(writer, "wait_closed"):
                try:
                    await writer.wait_closed()
                except Exception:
                    pass


@dataclass
class RecordedResponse:
    status: int
    reason: str | None
    headers: Message | None
    body: str | None
    wire_raw_bytes: bytes


@dataclass
class UpstreamResponse:
    """Upstream response context provided to transformers.

    Contains the parsed response data for inspection and transformation.
    The transformer can inspect any of these to decide how to transform
    the body.
    """

    status: int
    reason: str | None
    headers: Message | None
    body: bytes  # Decompressed body (raw bytes, not str)
    wire_raw_bytes: bytes  # Original wire format


@dataclass
class TransformResult:
    """Result of transforming an upstream response.

    The body field contains the new body bytes to send. When override_response
    is set, it takes precedence and completely replaces the upstream response.
    This is useful for fault injection scenarios like returning random errors.
    """

    body: bytes | None = None  # Transformed body (None = passthrough)
    override_response: HTTPResponse | None = None  # Replace entire response
    delay_before: float = 0.0  # Delay before sending
    drop_after: int | None = None  # Drop connection after N bytes


ResponseTransformer = Callable[
    [UpstreamResponse],
    TransformResult | Awaitable[TransformResult],
]


def fault_step_transformer(
    *steps: "FaultStep",
) -> ResponseTransformer:
    """Create a ResponseTransformer from FaultStep instances.

    This adapter allows reusing the fault injection primitives from
    server.py (ByteFlip, TruncateBody, Delay, DropConnection) with
    the proxy's response transformation.

    Example:
        from localstub.server import ByteFlip, Delay

        proxy = AsyncTLSInterceptProxy(
            default_mode="forward",
            response_transformer=fault_step_transformer(
                ByteFlip(offset=100, mask=0xFF),
                Delay(0.5),
            ),
        )
    """

    def transform(upstream: UpstreamResponse) -> TransformResult:
        body = upstream.body
        total_delay = 0.0
        drop_after: int | None = None

        for step in steps:
            result = step.apply(body)
            body = result.body
            total_delay += result.delay_before
            if drop_after is None and result.drop_after is not None:
                drop_after = result.drop_after

        return TransformResult(
            body=body,
            delay_before=total_delay,
            drop_after=drop_after,
        )

    return transform
