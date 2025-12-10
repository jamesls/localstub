from __future__ import annotations

import os
import ssl
import asyncio
import logging
import tempfile
from asyncio import transports
import gzip
from dataclasses import dataclass
from email.message import Message
from pathlib import Path
from typing import Optional, cast

import trustme

from localstub.http.response import (
    AsyncMultiResponseParser,
    AsyncResponseParser,
)
from localstub.http.request import (
    HTTPRequest,
    HTTPRequestReader,
)
from localstub.http.utils import headers_to_message
from localstub.server import AsyncHTTPTestServer


LOG = logging.getLogger(__name__)


def _wire_log(direction: str, data: bytes) -> None:
    """Log wire-level data with direction indicator."""
    LOG.debug("[%s] %r", direction, data)


def _close_log(direction: str, reason: str) -> None:
    """Log connection closure with direction and reason."""
    LOG.debug("[%s CLOSED] %s", direction, reason)


class _TrustMeCA:
    """Ephemeral CA backed by trustme; issues per-host server contexts."""

    def __init__(self) -> None:
        self._ca = trustme.CA()
        fd, path = tempfile.mkstemp(prefix="localstub-ca-", suffix=".pem")
        os.close(fd)
        Path(path).write_bytes(self._ca.cert_pem.bytes())
        self._ca_pem_path = Path(path)

    def issue_context(self, host: str) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_cert = self._ca.issue_server_cert(host)
        server_cert.configure_cert(context)
        context.set_alpn_protocols(["http/1.1"])
        context.options |= ssl.OP_NO_COMPRESSION
        return context

    def ca_pem_path(self) -> Path:
        return self._ca_pem_path


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
        ca: Optional[_TrustMeCA] = None,
        max_read: int = 8192,
        default_mode: str = "intercept",
        verify_upstream: bool = True,
        upstream_tls: bool = True,
    ) -> None:
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._server = server
        self._ca = ca or _TrustMeCA()
        self._max_read = max_read
        self._default_mode = default_mode
        self._verify_upstream = verify_upstream
        self._upstream_tls = upstream_tls

        self._listener: asyncio.base_events.Server | None = None
        self._host: str | None = None
        self._port: int | None = None
        self._client_tasks: set[asyncio.Task[None]] = set()

        # Recording queues for forwarded traffic
        self._recorded_requests: asyncio.Queue[HTTPRequest] = asyncio.Queue()
        self._recorded_responses: asyncio.Queue["RecordedResponse"] = (
            asyncio.Queue()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def address(self) -> tuple[str, int]:
        if self._host is None or self._port is None:
            raise RuntimeError("Proxy not started yet")
        return (self._host, self._port)

    @property
    def ca(self) -> _TrustMeCA:
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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

    async def _read_and_relay_responses(
        self,
        upstream_reader: asyncio.StreamReader,
        upstream_id: str,
        client_writer: asyncio.StreamWriter,
        client_id: str,
        request_method: str | None,
    ) -> RecordedResponse | None:
        """Read responses from upstream, relaying each to client.

        Handles 1xx informational responses by relaying them and continuing
        to read until a final (2xx+) response is received.

        Uses a single parser instance to handle cases where multiple responses
        (e.g., 100 Continue + 200 OK) arrive in a single buffer read.

        Returns the final response, or None if parsing failed.
        """
        multi_parser = AsyncMultiResponseParser(max_read=self._max_read)

        while True:
            parsed, wire_bytes = await multi_parser.next_response(
                upstream_reader, request_method
            )
            if parsed is None:
                return None

            _wire_log(f"{upstream_id} --> lstub", wire_bytes)

            # Relay the response to the client immediately
            _wire_log(f"lstub --> {client_id}", wire_bytes)
            client_writer.write(wire_bytes)
            await client_writer.drain()

            status = parsed.status_code or 0

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

        # Read and record the client's decrypted HTTP request first so we
        # preserve it even if the upstream cannot be reached.
        request_reader = HTTPRequestReader()
        request = await request_reader.read_request(
            client_reader, client_writer
        )
        if request is None or request.wire_raw_bytes is None:
            await self._send_and_close(
                client_writer, b"HTTP/1.1 400 Bad Request", client_id
            )
            return

        _wire_log(f"lstub <-- {client_id}", request.wire_raw_bytes)
        await self._recorded_requests.put(request)

        # Prepare upstream connection parameters after reading the request.
        use_tls = self._upstream_tls
        ssl_param: ssl.SSLContext | None = None
        server_hostname: str | None = None
        if use_tls:
            upstream_ctx = ssl.create_default_context()
            if not self._verify_upstream:
                upstream_ctx.check_hostname = False
                upstream_ctx.verify_mode = ssl.CERT_NONE
            ssl_param = upstream_ctx
            server_hostname = host

        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(
                host,
                port,
                ssl=ssl_param,
                server_hostname=server_hostname,
            )
        except Exception:
            # Could not connect upstream; send a 502 but keep the recorded
            # request available to callers for inspection.
            await self._send_and_close(
                client_writer, b"HTTP/1.1 502 Bad Gateway", client_id
            )
            return

        # Forward the raw request to upstream
        _wire_log(f"{upstream_id} <-- lstub", request.wire_raw_bytes)
        upstream_writer.write(request.wire_raw_bytes)
        await upstream_writer.drain()

        # Read responses from upstream, handling 1xx informational responses.
        # HTTP allows servers to send one or more 1xx responses before the
        # final response (e.g., 100 Continue before 200 OK).
        final_response = await self._read_and_relay_responses(
            upstream_reader,
            upstream_id,
            client_writer,
            client_id,
            request.method,
        )
        if final_response is None:
            await self._send_and_close(
                client_writer, b"HTTP/1.1 502 Bad Gateway", client_id
            )
            _close_log(f"{upstream_id} --> lstub", "failed to parse response")
            upstream_writer.close()
            await upstream_writer.wait_closed()
            return

        # Record only the final response
        await self._recorded_responses.put(final_response)

        # Close upstream connection
        _close_log(f"{upstream_id} --> lstub", "response received")
        upstream_writer.close()
        try:
            await upstream_writer.wait_closed()
        except Exception:
            pass

        # Close client connection
        _close_log(f"lstub --> {client_id}", "response relayed")
        client_writer.close()
        try:
            await client_writer.wait_closed()
        except Exception:
            pass

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
