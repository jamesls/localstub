from __future__ import annotations

import asyncio
import logging
import ssl
from asyncio import transports
from typing import TYPE_CHECKING, cast

from rich.markup import escape as rich_escape

if TYPE_CHECKING:
    from localstub.server import FaultStep

from localstub.forward import (
    ForwardError,
    Forwarder,
    ResponseTransformer,
    TransformResult,
    UpstreamResponse,
)
from localstub.http.response import (
    RecordedResponse,
)
from localstub.http.request import (
    AsyncRequestParser,
    HTTPRequest,
    ParsedRequest,
)
from localstub.ca import TLSProxyCA
from localstub.server import AsyncHTTPTestServer


LOG = logging.getLogger(__name__)


def _hexdump(data: bytes, bytes_per_line: int = 32) -> str:
    """Format bytes as tcpdump -X style hexdump.

    Args:
        data: Raw bytes to format.
        bytes_per_line: Number of bytes per line (default 32).

    Returns:
        Formatted hexdump string with Rich markup for consistent coloring.
    """
    lines = []
    # Calculate width for hex part: pairs * 4 chars + (pairs - 1) spaces
    num_pairs = bytes_per_line // 2
    hex_width = num_pairs * 4 + (num_pairs - 1)
    for offset in range(0, len(data), bytes_per_line):
        chunk = data[offset : offset + bytes_per_line]
        # Build hex pairs
        hex_pairs = []
        for i in range(0, len(chunk), 2):
            pair = chunk[i : i + 2]
            hex_pairs.append(pair.hex())
        hex_part = " ".join(hex_pairs)
        # Build ASCII part (escape for Rich markup safety)
        ascii_part = rich_escape(
            "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in chunk)
        )
        # Use Rich markup: dim for offset, grey70 for hex
        lines.append(
            f"[dim]0x{offset:04x}:[/dim]  "
            f"[grey39]{hex_part:<{hex_width}}[/grey39]  {ascii_part}"
        )
    return "\n".join(lines)


def _wire_log(direction: str, data: bytes) -> None:
    """Log wire-level data with direction indicator in hexdump format."""
    hexdump = _hexdump(data)
    # Escape direction (contains brackets), enable markup, disable highlighter
    escaped_dir = rich_escape(f"[{direction}]")
    LOG.debug(
        "%s\n%s",
        escaped_dir,
        hexdump,
        extra={"markup": True, "highlighter": None},
    )


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
        server: AsyncHTTPTestServer | None = None,
        ca: TLSProxyCA | None = None,
        max_read: int = 8192,
        default_mode: str = "intercept",
        verify_upstream: bool = True,
        upstream_tls: bool = True,
        response_transformer: ResponseTransformer | None = None,
        forwarder: Forwarder | None = None,
    ) -> None:
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._server = server
        self._ca = ca or TLSProxyCA()
        self._max_read = max_read
        self._default_mode = default_mode
        self._upstream_tls = upstream_tls
        if forwarder is None:
            forwarder = Forwarder(
                max_read=max_read,
                verify_upstream=verify_upstream,
                response_transformer=response_transformer,
                decompress_body=True,
                wire_log=_wire_log,
            )
        self._forwarder = forwarder

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

    async def _forward(
        self,
        host: str,
        port: int,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        client_id: str,
    ) -> None:
        upstream_id = f"{host}:{port}"

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

        upstream = await self._forwarder.connect_upstream(
            host,
            port,
            upstream_tls=self._upstream_tls,
        )
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

        try:
            if client_waiting:
                forwarded = await self._forwarder.forward_with_100_continue(
                    parser=parser,
                    parsed_headers=parsed,
                    header_wire_bytes=header_wire,
                    remaining_buffer=remaining,
                    client_reader=client_reader,
                    client_writer=client_writer,
                    upstream_reader=upstream_reader,
                    upstream_writer=upstream_writer,
                    request_method=parsed.method,
                    upstream_id=upstream_id,
                    client_id=client_id,
                )

                await self._record_request(
                    forwarded.parsed_request,
                    forwarded.request_wire_bytes,
                    client_writer,
                )

                if forwarded.error == ForwardError.REQUEST_PARSE_FAILED:
                    await self._send_and_close(
                        client_writer,
                        b"HTTP/1.1 400 Bad Request",
                        client_id,
                    )
                    return

                if forwarded.response is None:
                    await self._send_and_close(
                        client_writer,
                        b"HTTP/1.1 502 Bad Gateway",
                        client_id,
                    )
                    _close_log(
                        f"{upstream_id} --> lstub",
                        "failed to parse response",
                    )
                    return

                await self._recorded_responses.put(
                    forwarded.response.to_recorded_response()
                )
                _close_log(f"{upstream_id} --> lstub", "response received")
                _close_log(f"lstub --> {client_id}", "response relayed")
                client_writer.close()
                return

            final_parsed, full_wire = await parser.continue_parse_body(
                client_reader, remaining
            )
            if final_parsed is None:
                await self._send_and_close(
                    client_writer,
                    b"HTTP/1.1 400 Bad Request",
                    client_id,
                )
                return

            body_wire = full_wire[len(header_wire) :]
            if body_wire:
                _wire_log(f"lstub <-- {client_id}", body_wire)

            await self._record_request(final_parsed, full_wire, client_writer)
            _wire_log(f"{upstream_id} <-- lstub", full_wire)
            upstream_writer.write(full_wire)
            await upstream_writer.drain()

            final_response = await self._forwarder.read_and_relay_responses(
                upstream_reader=upstream_reader,
                client_writer=client_writer,
                request_method=final_parsed.method,
                upstream_id=upstream_id,
                client_id=client_id,
            )
            if final_response is None:
                await self._send_and_close(
                    client_writer,
                    b"HTTP/1.1 502 Bad Gateway",
                    client_id,
                )
                _close_log(
                    f"{upstream_id} --> lstub",
                    "failed to parse response",
                )
                return

            await self._recorded_responses.put(
                final_response.to_recorded_response()
            )
            _close_log(f"{upstream_id} --> lstub", "response received")
            _close_log(f"lstub --> {client_id}", "response relayed")
            client_writer.close()
        finally:
            upstream_writer.close()

    async def _record_request(
        self,
        parsed: ParsedRequest,
        wire_bytes: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Build and record an HTTPRequest."""
        request = HTTPRequest.from_parsed(parsed, wire_bytes, writer=writer)
        await self._recorded_requests.put(request)

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
