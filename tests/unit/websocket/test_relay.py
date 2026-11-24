from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from localstub.http.exchange import RecordedExchange
from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.websocket.frames import WebSocketFrame
from localstub.websocket.records import (
    RecordedWebSocket,
    WebSocketClose,
    WebSocketLog,
)
from localstub.websocket.relay import WebSocketRelay

MASK = b"\x01\x02\x03\x04"
TIMESTAMP = datetime(2026, 11, 24, tzinfo=UTC)
REQUEST = HTTPRequest(method="GET", target="/")
HANDSHAKE = RecordedExchange(
    request=RecordedHTTPRequest(
        request=REQUEST,
        as_received=REQUEST,
        wire_raw_bytes=b"",
        http_version="1.1",
    ),
    response=None,
    request_timestamp=TIMESTAMP,
    response_timestamp=None,
)


class FakeClock:
    def now(self) -> float:
        return 0.0


class FakeWriter:
    def __init__(self, error: OSError | None = None) -> None:
        self.written = bytearray()
        self.closed = False
        self.error = error

    def write(self, data: bytes) -> None:
        if self.error is not None:
            raise self.error
        self.written += data

    async def drain(self) -> None:
        await asyncio.sleep(0)

    def close(self) -> None:
        self.closed = True


def _log() -> WebSocketLog:
    return WebSocketLog(
        handshake=HANDSHAKE,
        subprotocol=None,
        compression=None,
        history_size=None,
    )


def _relay(
    *,
    log: WebSocketLog,
    client: asyncio.StreamReader,
    upstream: asyncio.StreamReader,
    client_writer: FakeWriter,
    upstream_writer: FakeWriter,
    timestamp: object = None,
    wire: list[tuple[str, bytes]] | None = None,
    **kwargs: bytes,
) -> WebSocketRelay:
    return WebSocketRelay(
        client_reader=client,
        client_writer=client_writer,
        upstream_reader=upstream,
        upstream_writer=upstream_writer,
        log=log,
        clock=FakeClock(),
        timestamp=timestamp or (lambda: TIMESTAMP),
        max_read=4,
        wire_log=None if wire is None else lambda d, b: wire.append((d, b)),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_relay_passes_bytes_unchanged_and_records_both_sides() -> None:
    log = _log()
    client, upstream = asyncio.StreamReader(), asyncio.StreamReader()
    to_client, to_upstream = FakeWriter(), FakeWriter()
    unmasked = WebSocketFrame.text("bad").encode()
    client_close = replace(WebSocketFrame.close(1000), mask_key=MASK).encode()
    server_close = WebSocketFrame.close(1001, "away").encode()
    wire: list[tuple[str, bytes]] = []
    relay = _relay(
        log=log,
        client=client,
        upstream=upstream,
        client_writer=to_client,
        upstream_writer=to_upstream,
        wire=wire,
        upstream_prefix=unmasked,
        client_prefix=server_close,
    )
    client.feed_data(client_close)

    task = asyncio.create_task(relay.run())
    await asyncio.sleep(0.01)
    client.feed_eof()
    await asyncio.wait_for(task, 1.0)

    recorded = RecordedWebSocket(log)
    assert bytes(to_upstream.written) == unmasked + client_close
    assert bytes(to_client.written) == server_close
    assert to_client.closed
    assert to_upstream.closed
    assert recorded.client_close == WebSocketClose(1000)
    assert recorded.server_close == WebSocketClose(1001, "away")
    assert [v.kind for v in recorded.violations] == ["unmasked_frame"]
    assert recorded.ended
    assert recorded.closed is None
    assert ("lstub <-- client", unmasked) in wire
    assert ("upstream <-- lstub", unmasked) in wire


@pytest.mark.asyncio
async def test_tap_failure_stops_recording_but_not_relaying(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def broken() -> datetime:
        raise RuntimeError("clock broke")

    log = _log()
    client, upstream = asyncio.StreamReader(), asyncio.StreamReader()
    to_upstream = FakeWriter()
    relay = _relay(
        log=log,
        client=client,
        upstream=upstream,
        client_writer=FakeWriter(),
        upstream_writer=to_upstream,
        timestamp=broken,
    )
    frame = replace(WebSocketFrame.text("a"), mask_key=MASK).encode()
    client.feed_data(frame + frame)
    client.feed_eof()

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(relay.run(), 1.0)

    assert bytes(to_upstream.written) == frame + frame
    assert RecordedWebSocket(log).frames == []
    assert caplog.text.count("tap failed") == 1


@pytest.mark.asyncio
async def test_relay_ends_when_a_side_fails() -> None:
    log = _log()
    client, upstream = asyncio.StreamReader(), asyncio.StreamReader()
    to_client = FakeWriter(ConnectionResetError())
    relay = _relay(
        log=log,
        client=client,
        upstream=upstream,
        client_writer=to_client,
        upstream_writer=FakeWriter(),
    )
    upstream.feed_data(WebSocketFrame.text("x").encode())

    await asyncio.wait_for(relay.run(), 1.0)

    assert to_client.closed
    assert RecordedWebSocket(log).ended
