from __future__ import annotations

import asyncio
import contextlib
import logging
import ssl
import tempfile
from asyncio import transports as asyncio_transports
from pathlib import Path
from typing import Optional, cast

try:
    import trustme
except ImportError as exc:  # pragma: no cover - handled in tests
    raise RuntimeError(
        "trustme is required for AsyncTLSInterceptProxy; "
        "install the 'tls' extra or add trustme to your dependencies."
    ) from exc

from .server import AsyncHTTPTestServer

LOG = logging.getLogger(__name__)


class _TrustMeCA:
    """Ephemeral CA backed by trustme; issues per-host server contexts."""

    def __init__(self) -> None:
        self._ca = trustme.CA()
        fd, path = tempfile.mkstemp(prefix="localstub-ca-", suffix=".pem")
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
    ) -> None:
        self._listen_host = listen_host
        self._listen_port = listen_port
        self._server = server
        self._ca = ca or _TrustMeCA()
        self._max_read = max_read
        self._default_mode = default_mode
        self._verify_upstream = verify_upstream

        self._listener: asyncio.base_events.Server | None = None
        self._host: str | None = None
        self._port: int | None = None

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

    async def start(self) -> None:
        if self._listener is not None:
            return
        self._listener = await asyncio.start_server(
            self._handle_client,
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
        try:
            connect_host, connect_port = await self._parse_connect(reader)
            if connect_host is None or connect_port is None:
                await self._send_and_close(writer, b"HTTP/1.1 400 Bad Request")
                return

            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()

            tls_reader, tls_writer = await self._upgrade_to_tls(
                writer, connect_host
            )

            if self._default_mode == "forward":
                await self._forward(
                    connect_host,
                    connect_port,
                    tls_reader,
                    tls_writer,
                )
                return

            if self._server is None:
                await self._send_and_close(
                    tls_writer, b"HTTP/1.1 502 Bad Gateway"
                )
                return

            await self._server._handle_client(tls_reader, tls_writer)
        except Exception:
            LOG.exception("TLS proxy error")
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _parse_connect(
        self, reader: asyncio.StreamReader
    ) -> tuple[str | None, int | None]:
        line = await reader.readline()
        if not line:
            return None, None
        try:
            req_line = line.decode("ascii", errors="replace").strip()
            parts = req_line.split(" ")
            if len(parts) < 3:
                return None, None
            method, target, _version = parts[0], parts[1], parts[2]
            if method.upper() != "CONNECT":
                return None, None
            if ":" in target:
                host, port_str = target.split(":", 1)
            else:
                host, port_str = target, "443"
            port = int(port_str)
        except Exception:
            return None, None

        # Consume remaining headers up to blank line
        while True:
            header_line = await reader.readline()
            if not header_line:
                break
            if header_line in (b"\r\n", b"\n"):
                break
        return host, port

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
        tls_protocol = asyncio.StreamReaderProtocol(tls_reader)

        tls_transport = await loop.start_tls(
            transport,
            tls_protocol,
            ssl_context,
            server_side=True,
            ssl_handshake_timeout=10.0,
        )

        tls_writer = asyncio.StreamWriter(
            cast(asyncio_transports.WriteTransport, tls_transport),
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
    ) -> None:
        upstream_ssl: ssl.SSLContext | bool | None
        if self._verify_upstream:
            upstream_ssl = ssl.create_default_context()
        else:
            upstream_ssl = None

        upstream_reader, upstream_writer = await asyncio.open_connection(
            host,
            port,
            ssl=upstream_ssl,
            server_hostname=host if upstream_ssl else None,
        )

        async def relay(
            reader: asyncio.StreamReader, writer: asyncio.StreamWriter
        ) -> None:
            try:
                while True:
                    data = await reader.read(8192)
                    if not data:
                        break
                    writer.write(data)
                    await writer.drain()
            except Exception:
                return
            finally:
                try:
                    writer.write_eof()
                except Exception:
                    writer.close()

        relay_to_upstream = asyncio.create_task(
            relay(client_reader, upstream_writer)
        )
        relay_to_client = asyncio.create_task(
            relay(upstream_reader, client_writer)
        )

        try:
            await asyncio.wait(
                {relay_to_upstream, relay_to_client},
                return_when=asyncio.ALL_COMPLETED,
            )
        finally:
            relay_to_upstream.cancel()
            relay_to_client.cancel()
            with contextlib.suppress(Exception):
                upstream_writer.close()
                await upstream_writer.wait_closed()
            with contextlib.suppress(Exception):
                client_writer.close()
                await client_writer.wait_closed()

    async def _send_and_close(
        self,
        writer: asyncio.StreamWriter,
        status_line: bytes,
    ) -> None:
        try:
            writer.write(status_line + b"\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()
            if hasattr(writer, "wait_closed"):
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
