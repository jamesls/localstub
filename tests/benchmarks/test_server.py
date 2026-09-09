from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import pytest
from pytest_codspeed import BenchmarkFixture

from localstub import AsyncHTTPTestServer
from localstub.recording import DEFAULT_MAX_CONNECTION_BYTES, BoundedByteBuffer
from localstub.server import RecordingStreamWriter

RESPONSE_OBJECT = {"ok": True, "items": [1, 2, 3]}
RESPONSE_BODY = json.dumps(RESPONSE_OBJECT).encode("utf-8")
LARGE_BODY = b"b" * (64 * 1024)
BODY_PIECE = b"p" * 8192
BODY_PIECE_COUNT = 128
SMALL_PIECE = b"s" * 16
SMALL_PIECE_COUNT = 4096
REQUEST = (
    b"GET /v1/items?page=2 HTTP/1.1\r\n"
    b"Host: localhost\r\n"
    b"User-Agent: localstub-bench/1.0\r\n"
    b"Accept: application/json\r\n"
    b"\r\n"
)

type Connection = tuple[asyncio.StreamReader, asyncio.StreamWriter]


@pytest.fixture
def connection(loop: asyncio.AbstractEventLoop) -> Iterator[Connection]:
    # One keep-alive connection is opened outside the measured region so
    # each sample is a single request/response exchange through the whole
    # server stack, without connect or accept costs.
    server = AsyncHTTPTestServer()
    server.set_json_response(RESPONSE_OBJECT)
    loop.run_until_complete(server.start())
    reader, writer = loop.run_until_complete(
        asyncio.open_connection(server.host, server.port)
    )
    try:
        yield reader, writer
    finally:
        writer.close()
        loop.run_until_complete(writer.wait_closed())
        loop.run_until_complete(server.aclose())


async def exchange(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> tuple[bytes, bytes]:
    # A raw client keeps HTTP client libraries out of the measurement.
    # The response is static, so its body length is known up front.
    writer.write(REQUEST)
    await writer.drain()
    head = await reader.readuntil(b"\r\n\r\n")
    body = await reader.readexactly(len(RESPONSE_BODY))
    return head, body


def test_server_roundtrip_static_json(
    benchmark: BenchmarkFixture,
    loop: asyncio.AbstractEventLoop,
    connection: Connection,
) -> None:
    reader, writer = connection

    def exchange_once() -> tuple[bytes, bytes]:
        return loop.run_until_complete(exchange(reader, writer))

    head, body = benchmark(exchange_once)

    assert head.startswith(b"HTTP/1.1 200 OK\r\n")
    assert body == RESPONSE_BODY


class _NullTransport(asyncio.Transport):
    # Absorbs writes so the writer benchmarks measure recording, not I/O.

    def __init__(self) -> None:
        super().__init__()
        self._closed = False

    def write(self, data: bytes | bytearray | memoryview) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    def is_closing(self) -> bool:
        return self._closed


@pytest.fixture
def recording_writer(
    loop: asyncio.AbstractEventLoop,
) -> Iterator[RecordingStreamWriter]:
    # The connection buffer persists across samples, as it does for a
    # long-lived connection, so it reaches capacity and churns.
    writer = asyncio.StreamWriter(
        _NullTransport(), asyncio.Protocol(), None, loop
    )
    sent = BoundedByteBuffer(DEFAULT_MAX_CONNECTION_BYTES)
    try:
        yield RecordingStreamWriter(writer, sent)
    finally:
        writer.close()


def test_recording_writer_single_write_response(
    benchmark: BenchmarkFixture,
    recording_writer: RecordingStreamWriter,
) -> None:
    # A static response goes out in one write; recording it should not
    # copy the body.
    def respond() -> bytes:
        recording_writer.start_response()
        recording_writer.write(LARGE_BODY)
        return recording_writer.bytes_sent

    sent = benchmark(respond)

    assert sent is LARGE_BODY


def test_recording_writer_body_in_many_writes(
    benchmark: BenchmarkFixture,
    recording_writer: RecordingStreamWriter,
) -> None:
    # A throttled or chunked body goes out piecewise; the recording is
    # joined once when the completed response is requested.
    def respond() -> bytes:
        recording_writer.start_response()
        for _ in range(BODY_PIECE_COUNT):
            recording_writer.write(BODY_PIECE)
        return recording_writer.bytes_sent

    sent = benchmark(respond)

    assert sent == BODY_PIECE * BODY_PIECE_COUNT


def test_recording_writer_body_in_small_writes(
    benchmark: BenchmarkFixture,
    recording_writer: RecordingStreamWriter,
) -> None:
    # A tightly throttled body goes out in many small writes; recording
    # coalesces them so the join at the end touches few pieces.
    def respond() -> bytes:
        recording_writer.start_response()
        for _ in range(SMALL_PIECE_COUNT):
            recording_writer.write(SMALL_PIECE)
        return recording_writer.bytes_sent

    sent = benchmark(respond)

    assert sent == SMALL_PIECE * SMALL_PIECE_COUNT
