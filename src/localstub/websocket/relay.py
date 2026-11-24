"""Relay a WebSocket byte for byte and record both directions.

``WebSocketRelay`` runs one pump per direction.  A pump reads, writes
the same bytes to the other side, drains, and feeds them to a tap.
Nothing is re-framed: mask keys, fragmentation, length encodings and
protocol violations reach the other side exactly as they were sent.

A tap decodes its copy of the bytes into the shared record.  It never
writes to a socket and never acts on a violation, and a tap failure
never stops its pump.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Protocol

from localstub.clock import Clock
from localstub.http.stream import ByteStream
from localstub.websocket.assembly import MessageAssembler
from localstub.websocket.frames import FrameDecoder
from localstub.websocket.records import (
    RecordedWebSocketFrame,
    Sender,
    WebSocketLog,
)

LOG = logging.getLogger(__name__)

type WireLog = Callable[[str, bytes], None]


class RelayWriter(Protocol):
    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...

    def close(self) -> None: ...


class _Tap:
    """Decode one direction's bytes into the shared log."""

    def __init__(
        self,
        *,
        sender: Sender,
        log: WebSocketLog,
        clock: Clock,
        timestamp: Callable[[], datetime],
        unknown_extensions: bool,
    ) -> None:
        self._sender = sender
        self._log = log
        self._clock = clock
        self._timestamp = timestamp
        self._decoder = FrameDecoder()
        self._assembler = MessageAssembler(
            sender=sender,
            compression=log.compression,
            strict=False,
            max_message_bytes=None,
            unknown_extensions=unknown_extensions,
        )
        self._broken = False

    def feed(self, data: bytes) -> None:
        if self._broken:
            return
        try:
            self._feed(data)
        except Exception:
            self._broken = True
            LOG.exception("WebSocket relay tap failed; recording stopped")

    def _feed(self, data: bytes) -> None:
        log = self._log
        for decoded in self._decoder.feed(data):
            recorded = RecordedWebSocketFrame(
                frame=decoded.frame,
                sender=self._sender,
                wire_raw_bytes=decoded.wire_raw_bytes,
                timestamp=self._timestamp(),
                monotonic=self._clock.now(),
            )
            log.add_frame(recorded)
            result = self._assembler.receive(recorded, decoded.header)
            for violation in result.violations:
                log.add_violation(violation)
            for message in result.messages:
                log.add_message(message)
            if result.close is not None:
                log.set_close(self._sender, result.close)


class WebSocketRelay:
    """Pass a switched connection through unchanged, recording it.

    ``upstream_prefix`` and ``client_prefix`` are bytes already read
    past the handshake on each side, sent before the pumps start.
    """

    def __init__(
        self,
        *,
        client_reader: ByteStream,
        client_writer: RelayWriter,
        upstream_reader: ByteStream,
        upstream_writer: RelayWriter,
        log: WebSocketLog,
        clock: Clock,
        timestamp: Callable[[], datetime],
        max_read: int,
        unknown_extensions: bool = False,
        upstream_prefix: bytes = b"",
        client_prefix: bytes = b"",
        wire_log: WireLog | None = None,
        client_id: str = "client",
        upstream_id: str = "upstream",
    ) -> None:
        self._client_reader = client_reader
        self._client_writer = client_writer
        self._upstream_reader = upstream_reader
        self._upstream_writer = upstream_writer
        self._log = log
        self._max_read = max_read
        self._upstream_prefix = upstream_prefix
        self._client_prefix = client_prefix
        self._wire_log = wire_log
        self._client_id = client_id
        self._upstream_id = upstream_id
        self._client_tap = _Tap(
            sender="client",
            log=log,
            clock=clock,
            timestamp=timestamp,
            unknown_extensions=unknown_extensions,
        )
        self._upstream_tap = _Tap(
            sender="server",
            log=log,
            clock=clock,
            timestamp=timestamp,
            unknown_extensions=unknown_extensions,
        )

    async def run(self) -> None:
        """Relay until either side ends, then close both and return."""
        to_upstream = asyncio.create_task(
            self._pump(
                self._client_reader,
                self._upstream_writer,
                self._client_tap,
                self._upstream_prefix,
                (
                    f"lstub <-- {self._client_id}",
                    f"{self._upstream_id} <-- lstub",
                ),
            )
        )
        to_client = asyncio.create_task(
            self._pump(
                self._upstream_reader,
                self._client_writer,
                self._upstream_tap,
                self._client_prefix,
                (
                    f"{self._upstream_id} --> lstub",
                    f"lstub --> {self._client_id}",
                ),
            )
        )
        pumps = (to_upstream, to_client)
        try:
            await asyncio.wait(pumps, return_when="FIRST_COMPLETED")
        finally:
            for pump in pumps:
                pump.cancel()
            await asyncio.gather(*pumps, return_exceptions=True)
            for writer in (self._client_writer, self._upstream_writer):
                writer.close()
            self._log.end(None)

    async def _pump(
        self,
        reader: ByteStream,
        writer: RelayWriter,
        tap: _Tap,
        prefix: bytes,
        labels: tuple[str, str],
    ) -> None:
        data = prefix
        try:
            while True:
                if data:
                    self._log_wire(labels, data)
                    writer.write(data)
                    await writer.drain()
                    tap.feed(data)
                data = await reader.read(self._max_read)
                if not data:
                    return
        except OSError:
            LOG.debug("WebSocket relay side failed", exc_info=True)

    def _log_wire(self, labels: tuple[str, str], data: bytes) -> None:
        wire_log = self._wire_log
        if wire_log is None:
            return
        wire_log(labels[0], data)
        wire_log(labels[1], data)
