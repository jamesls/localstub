from __future__ import annotations

import asyncio
import logging
import ssl
from asyncio import transports
from datetime import datetime
from typing import TYPE_CHECKING, Self, cast

from rich.markup import escape as rich_escape

if TYPE_CHECKING:
    from localstub.server import FaultStep

from localstub.ca import TLSProxyCA
from localstub.forward import (
    ForwardError,
    RawForwarder,
    ResponseTransformer,
    TransformContext,
    TransformResult,
    response_allows_keep_alive,
)
from localstub.http.exchange import RecordedExchange
from localstub.http.request import (
    AsyncRequestParser,
    ParsedRequest,
    RecordedHTTPRequest,
)
from localstub.http.response import (
    RecordedHTTPResponse,
)
from localstub.recording import (
    DEFAULT_RECORDING_BUFFER_SIZE,
    TrafficRecorder,
)
from localstub.server import AsyncHTTPTestServer

LOG = logging.getLogger(__name__)
_SHUTDOWN_DRAIN_TURNS = 8


def _pause_listener_accepts(listener: asyncio.Server) -> None:
    loop = asyncio.get_running_loop()
    try:
        for sock in listener.sockets:
            loop.remove_reader(sock.fileno())
    except NotImplementedError:
        listener.close()


async def _drain_pending_accepts() -> None:
    for _ in range(_SHUTDOWN_DRAIN_TURNS):
        await asyncio.sleep(0)


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
    if not LOG.isEnabledFor(logging.DEBUG):
        # Formatting a hexdump is expensive and this is on the hot path for
        # every byte proxied, so skip the work when it would be discarded.
        return
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
        forwarder: RawForwarder | None = None,
        recording_buffer_size: int | None = DEFAULT_RECORDING_BUFFER_SIZE,
        recorder: TrafficRecorder | None = None,
    ) -> None:
        # Recording state for forwarded traffic.  Bounded so memory stays
        # flat when a consumer never drains a stream (e.g. the CLI only
        # reads exchanges).  Created first so an invalid buffer size fails
        # before the CA default resolution generates keys and writes files.
        self._recorder = recorder or TrafficRecorder(recording_buffer_size)

        self._listen_host = listen_host
        self._listen_port = listen_port
        self._server = server
        self._ca = ca or TLSProxyCA()
        self._max_read = max_read
        self._default_mode = default_mode
        self._upstream_tls = upstream_tls
        if forwarder is None:
            forwarder = RawForwarder(
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
        self._closing = False
        self._client_writers: set[asyncio.StreamWriter] = set()
        self._client_tasks: set[asyncio.Task[None]] = set()

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

    async def next_request(
        self, timeout: float | None = None
    ) -> RecordedHTTPRequest:
        return await self._recorder.next_request(timeout)

    async def next_response(
        self, timeout: float | None = None
    ) -> RecordedHTTPResponse:
        return await self._recorder.next_response(timeout)

    async def next_exchange(
        self, timeout: float | None = None
    ) -> RecordedExchange:
        return await self._recorder.next_exchange(timeout)

    def next_exchange_nowait(self) -> RecordedExchange | None:
        return self._recorder.next_exchange_nowait()

    @property
    def dropped_requests(self) -> int:
        """Requests evicted unread from the next_request() buffer."""
        return self._recorder.dropped_requests

    @property
    def dropped_responses(self) -> int:
        """Responses evicted unread from the next_response() buffer."""
        return self._recorder.dropped_responses

    @property
    def dropped_exchanges(self) -> int:
        """Exchanges evicted unread from the next_exchange() buffer."""
        return self._recorder.dropped_exchanges

    async def start(self) -> None:
        if self._listener is not None:
            return
        self._closing = False
        self._listener = await asyncio.start_server(
            self._client_connected,
            self._listen_host,
            self._listen_port,
        )
        assert self._listener.sockets
        sockname = self._listener.sockets[0].getsockname()
        self._host, self._port = sockname[0], sockname[1]

    async def aclose(self) -> None:
        listener = self._listener
        if listener is None:
            return
        self._closing = True
        # aclose() may be called from inside a client task (e.g. a request
        # handler shutting down the proxy); waiting on that task would
        # deadlock, so it is excluded from the drain.
        caller = asyncio.current_task()
        shutdown_task = asyncio.create_task(
            self._finish_close(listener, exclude=caller)
        )
        try:
            await asyncio.shield(shutdown_task)
        except asyncio.CancelledError:
            await shutdown_task
            raise
        finally:
            listener.close()
            if self._listener is listener:
                self._listener = None
            self._closing = False

    async def _finish_close(
        self,
        listener: asyncio.Server,
        exclude: asyncio.Task[object] | None,
    ) -> None:
        _pause_listener_accepts(listener)
        await _drain_pending_accepts()
        listener.close()

        client_tasks = tuple(
            task for task in self._client_tasks if task is not exclude
        )
        pending: set[asyncio.Task[None]] = set()
        if client_tasks:
            _, pending = await asyncio.wait(client_tasks, timeout=1.0)

        writers = tuple(self._client_writers)
        for writer in writers:
            writer.close()
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for writer in writers:
            try:
                await writer.wait_closed()
            except Exception:
                LOG.debug(
                    "Failed to close TLS proxy client writer",
                    exc_info=True,
                )

        await listener.wait_closed()
        self._client_tasks.clear()
        self._client_writers.clear()

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        active_writer = writer
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
                _close_log(f"lstub --> {client_id}", "TLS handshake failed")
                return
            active_writer = self._replace_client_writer(writer, tls_writer)

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
            _close_log(f"lstub --> {client_id}", "task cancelled")
            # Cancellation must not wait for the peer's TLS shutdown reply.
            writer.transport.abort()
            raise
        except Exception:
            # Any other exception is unexpected; keep the traceback to aid
            # debugging.
            LOG.exception("TLS proxy error")
            _close_log(f"lstub --> {client_id}", "unexpected error")
        finally:
            try:
                active_writer.close()
                close_task = asyncio.create_task(active_writer.wait_closed())
                try:
                    # aclose() also awaits this protocol's close waiter.
                    await asyncio.shield(close_task)
                except asyncio.CancelledError:
                    writer.transport.abort()
                    await asyncio.gather(close_task, return_exceptions=True)
                    raise
            except OSError:
                LOG.debug(
                    "Failed to close TLS proxy client writer",
                    exc_info=True,
                )
            finally:
                # Retain the original writer until TLS shutdown finishes.
                # Close its transport if shutdown fails or is cancelled.
                writer.close()
                self._client_writers.discard(writer)
                self._client_writers.discard(active_writer)

    def _replace_client_writer(
        self,
        current: asyncio.StreamWriter,
        replacement: asyncio.StreamWriter,
    ) -> asyncio.StreamWriter:
        self._client_writers.add(replacement)
        self._client_writers.discard(current)
        return replacement

    def _client_connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        if self._closing:
            writer.close()
            return
        self._client_writers.add(writer)
        task = asyncio.create_task(self._handle_client(reader, writer))
        self._client_tasks.add(task)

        def release_client(done_task: asyncio.Task[None]) -> None:
            self._client_tasks.discard(done_task)
            if writer in self._client_writers:
                writer.close()
                self._client_writers.discard(writer)

        task.add_done_callback(release_client)

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
            LOG.debug("Failed to parse CONNECT target", exc_info=True)
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

        while True:
            parser = AsyncRequestParser(max_read=self._max_read)
            parsed, header_wire, remaining = await parser.parse_headers(
                client_reader
            )
            if parsed is None:
                # Bytes received but unparseable is a malformed request; an
                # empty read is a clean EOF from a client that closed a
                # persistent connection, so just close our side.
                if header_wire:
                    await self._send_and_close(
                        client_writer, b"HTTP/1.1 400 Bad Request", client_id
                    )
                else:
                    self._close_client(client_writer, client_id)
                return

            _wire_log(f"lstub <-- {client_id}", header_wire)

            keep_alive = await self._forward_one_request(
                host=host,
                port=port,
                upstream_id=upstream_id,
                client_id=client_id,
                parser=parser,
                parsed=parsed,
                header_wire=header_wire,
                remaining=remaining,
                client_reader=client_reader,
                client_writer=client_writer,
            )

            if client_writer.is_closing():
                # An error response or fault injection already tore the
                # connection down; there is nothing left to keep alive.
                return
            if not keep_alive:
                self._close_client(client_writer, client_id)
                return

    async def _forward_one_request(
        self,
        *,
        host: str,
        port: int,
        upstream_id: str,
        client_id: str,
        parser: AsyncRequestParser,
        parsed: ParsedRequest,
        header_wire: bytes,
        remaining: bytearray,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
    ) -> bool:
        """Forward a single request/response exchange.

        Returns whether the client connection may be reused for a
        subsequent request. Error and fault-injection paths close the
        client writer themselves and return ``False``.
        """
        # Check for Expect: 100-continue with client actually waiting.
        expect_hdr = {k.lower(): v for k, v in parsed.headers}.get(
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
            request, request_timestamp = self._record_request(
                parsed, wire_bytes, client_writer
            )
            await self._record_failure_and_close(
                request,
                request_timestamp,
                client_writer,
                b"HTTP/1.1 502 Bad Gateway",
                client_id,
            )
            return False
        upstream_reader, upstream_writer = upstream

        try:
            if client_waiting:
                return await self._forward_100_continue_request(
                    parser=parser,
                    parsed=parsed,
                    header_wire=header_wire,
                    remaining=remaining,
                    client_reader=client_reader,
                    client_writer=client_writer,
                    upstream_reader=upstream_reader,
                    upstream_writer=upstream_writer,
                    upstream_id=upstream_id,
                    client_id=client_id,
                )

            final_parsed, full_wire = await parser.continue_parse_body(
                client_reader, remaining
            )
            if final_parsed is None:
                await self._send_and_close(
                    client_writer,
                    b"HTTP/1.1 400 Bad Request",
                    client_id,
                )
                return False

            body_wire = full_wire[len(header_wire) :]
            if body_wire:
                _wire_log(f"lstub <-- {client_id}", body_wire)

            request, request_timestamp = self._record_request(
                final_parsed, full_wire, client_writer
            )
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
                await self._record_failure_and_close(
                    request,
                    request_timestamp,
                    client_writer,
                    b"HTTP/1.1 502 Bad Gateway",
                    client_id,
                )
                _close_log(
                    f"{upstream_id} --> lstub",
                    "failed to parse response",
                )
                return False

            response = final_response.to_recorded_response()
            self._recorder.record_exchange(
                request=request,
                response=response,
                request_timestamp=request_timestamp,
            )
            _close_log(f"{upstream_id} --> lstub", "response received")
            _close_log(f"lstub --> {client_id}", "response relayed")
            return response_allows_keep_alive(request, final_response)
        finally:
            upstream_writer.close()
            try:
                await upstream_writer.wait_closed()
            except OSError:
                LOG.debug(
                    "Failed to close upstream writer",
                    exc_info=True,
                )

    async def _forward_100_continue_request(
        self,
        *,
        parser: AsyncRequestParser,
        parsed: ParsedRequest,
        header_wire: bytes,
        remaining: bytearray,
        client_reader: asyncio.StreamReader,
        client_writer: asyncio.StreamWriter,
        upstream_reader: asyncio.StreamReader,
        upstream_writer: asyncio.StreamWriter,
        upstream_id: str,
        client_id: str,
    ) -> bool:
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

        request, request_timestamp = self._record_request(
            forwarded.parsed_request,
            forwarded.request_wire_bytes,
            client_writer,
        )

        if forwarded.error == ForwardError.REQUEST_PARSE_FAILED:
            await self._record_failure_and_close(
                request,
                request_timestamp,
                client_writer,
                b"HTTP/1.1 400 Bad Request",
                client_id,
            )
            return False

        if forwarded.response is None:
            await self._record_failure_and_close(
                request,
                request_timestamp,
                client_writer,
                b"HTTP/1.1 502 Bad Gateway",
                client_id,
            )
            _close_log(
                f"{upstream_id} --> lstub",
                "failed to parse response",
            )
            return False

        response = forwarded.response.to_recorded_response()
        self._recorder.record_exchange(
            request=request,
            response=response,
            request_timestamp=request_timestamp,
        )
        _close_log(f"{upstream_id} --> lstub", "response received")
        _close_log(f"lstub --> {client_id}", "response relayed")
        if not forwarded.request_body_consumed:
            # An early final response leaves the declared request body
            # unread; a client that stopped waiting may still send it,
            # so later bytes have ambiguous framing (RFC 9112 §9.6).
            return False
        return response_allows_keep_alive(request, forwarded.response)

    def _close_client(
        self,
        writer: asyncio.StreamWriter,
        client_id: str,
    ) -> None:
        _close_log(f"lstub --> {client_id}", "connection closed")
        writer.close()

    def _record_request(
        self,
        parsed: ParsedRequest,
        wire_bytes: bytes,
        writer: asyncio.StreamWriter,
    ) -> tuple[RecordedHTTPRequest, datetime]:
        """Build and record a RecordedHTTPRequest."""
        request = RecordedHTTPRequest.from_parsed(
            parsed, wire_bytes, writer=writer
        )
        request_timestamp, _ = self._recorder.record_request(request)
        return request, request_timestamp

    async def _record_failure_and_close(
        self,
        request: RecordedHTTPRequest,
        request_timestamp: datetime,
        writer: asyncio.StreamWriter,
        status_line: bytes,
        client_id: str,
    ) -> None:
        self._recorder.record_exchange(
            request=request,
            response=None,
            request_timestamp=request_timestamp,
        )
        await self._send_and_close(writer, status_line, client_id)

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
                    LOG.debug(
                        "Failed to close client writer",
                        exc_info=True,
                    )


def fault_step_transformer(
    *steps: FaultStep,
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

    def transform(context: TransformContext) -> TransformResult:
        body = context.body
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
