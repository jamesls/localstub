from __future__ import annotations

import asyncio
from datetime import UTC, datetime

from pytest_codspeed import BenchmarkFixture

from localstub.clock import MonotonicClock
from localstub.http.exchange import (
    CloseReason,
    ConnectionClosed,
    RecordedExchange,
)
from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.websocket.frames import FrameDecoder, Opcode, WebSocketFrame
from localstub.websocket.records import WebSocketLog
from localstub.websocket.session import SessionOptions, WebSocketSession

MASK = b"\x37\xfa\x21\x3d"
LARGE_FRAME = WebSocketFrame(
    Opcode.BINARY, bytes(range(256)) * 4096, mask_key=MASK
).encode()
ECHO_COUNT = 1000
SMALL_FRAME = WebSocketFrame(
    Opcode.TEXT, b'{"op": "ping"}', mask_key=MASK
).encode()
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


class _Writer:
    def __init__(self) -> None:
        self.written = 0

    def write(self, data: bytes | memoryview) -> None:
        self.written += len(data)

    async def drain(self) -> None:
        return None


def test_decode_1_mib_masked_frame(benchmark: BenchmarkFixture) -> None:
    # Unmasking is the per-byte cost of every client frame, so a large
    # upload through a session lives on this path.
    def decode() -> int:
        [decoded] = FrameDecoder().feed(LARGE_FRAME)
        return len(decoded.frame.payload)

    assert benchmark(decode) == 1024 * 1024


def test_echo_round_trip_small_messages(
    benchmark: BenchmarkFixture, loop: asyncio.AbstractEventLoop
) -> None:
    # A handler echoing small messages exercises the reader loop,
    # assembly, recording, and the send path once per message.
    async def echo(ws: WebSocketSession) -> None:
        for _ in range(ECHO_COUNT):
            await ws.send(await ws.receive())
        # Ending without a closing handshake keeps close_timeout, which
        # waits for a client that is not there, out of the measurement.
        await ws.drop()

    def close(reason: CloseReason, reset: bool) -> ConnectionClosed:
        return ConnectionClosed(
            client=None,
            reason=reason,
            phase="websocket",
            reset=reset,
            requests_completed=1,
            bytes_read=0,
            bytes_consumed=0,
            bytes_written=0,
            timestamp=TIMESTAMP,
        )

    async def run() -> int:
        reader = asyncio.StreamReader()
        reader.feed_data(SMALL_FRAME * ECHO_COUNT)
        writer = _Writer()
        session = WebSocketSession(
            reader=reader,
            writer=writer,
            log=WebSocketLog(
                handshake=HANDSHAKE,
                subprotocol=None,
                compression=None,
                history_size=256,
            ),
            close=close,
            handler=echo,
            # Every frame arrives in one read, before the handler runs,
            # so the receive queue must hold them all.
            options=SessionOptions(queue_size=None),
            sleep=asyncio.sleep,
            clock=MonotonicClock(),
            timestamp=lambda: TIMESTAMP,
        )
        await session.run()
        return writer.written

    written = benchmark(lambda: loop.run_until_complete(run()))

    expected_frames = len(WebSocketFrame.text('{"op": "ping"}').encode())
    assert written == expected_frames * ECHO_COUNT
