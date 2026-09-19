from __future__ import annotations

import asyncio
import json
import selectors
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest
from pytest_codspeed import BenchmarkFixture

from localstub import (
    AsyncHTTPTestServer,
    HTTPResponse,
    ResponderContext,
    ResponderNext,
    ResponseSpec,
)
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


class _ReadySelector(selectors.SelectSelector):
    def select(
        self, timeout: float | None = None
    ) -> list[tuple[selectors.SelectorKey, int]]:
        # In-memory exchanges only need ready callbacks. Fail if a sample
        # would block, rather than polling the OS or spinning forever.
        assert timeout == 0, "in-memory benchmark is waiting for I/O"
        return []


class _InMemoryLoop(asyncio.SelectorEventLoop):
    def time(self) -> float:
        return 0.0


@dataclass(frozen=True)
class _FixedClock[T]:
    value: T

    def now(self) -> T:
        return self.value


class _MemoryTransport(asyncio.Transport):
    def __init__(
        self,
        peer_reader: asyncio.StreamReader,
        protocol: asyncio.StreamReaderProtocol,
    ) -> None:
        super().__init__({"peername": ("127.0.0.1", 54321)})
        self._peer_reader = peer_reader
        self._protocol = protocol
        self._closed = False

    def write(self, data: bytes | bytearray | memoryview) -> None:
        self._peer_reader.feed_data(data)

    def close(self) -> None:
        self._closed = True
        self._peer_reader.feed_eof()
        self._protocol.connection_lost(None)

    def is_closing(self) -> bool:
        return self._closed


@pytest.fixture
def routed_loop() -> Iterator[asyncio.AbstractEventLoop]:
    # Loop setup stays outside measurement; samples perform no socket,
    # selector, or clock I/O.
    loop = _InMemoryLoop(selector=_ReadySelector())
    try:
        yield loop
    finally:
        loop.close()


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


@pytest.fixture
def routed_server() -> AsyncHTTPTestServer:
    response = HTTPResponse.json(RESPONSE_OBJECT)

    async def handler(_: ResponderContext) -> HTTPResponse:
        return response

    async def pass_through(
        _: ResponderContext, call_next: ResponderNext
    ) -> ResponseSpec:
        return await call_next()

    server = AsyncHTTPTestServer(
        clock=_FixedClock(0.0),
        timestamp_provider=_FixedClock(datetime(2026, 1, 1, tzinfo=UTC)),
    )
    # A route miss returns the same body length, so exchange() completes
    # and the status assertion catches the miss.
    server.set_json_response(RESPONSE_OBJECT, status=404)
    server.add_route("GET", "/v1/items?page=2", handler)
    for _ in range(3):
        server.use(pass_through)
    return server


@pytest.fixture
def routed_connection(
    routed_loop: asyncio.AbstractEventLoop,
    routed_server: AsyncHTTPTestServer,
) -> Iterator[Connection]:
    reader = asyncio.StreamReader(loop=routed_loop)
    server_reader = asyncio.StreamReader(loop=routed_loop)
    protocol = asyncio.StreamReaderProtocol(reader, loop=routed_loop)
    server_protocol = asyncio.StreamReaderProtocol(
        server_reader, loop=routed_loop
    )
    writer = asyncio.StreamWriter(
        _MemoryTransport(server_reader, protocol),
        protocol,
        reader,
        routed_loop,
    )
    server_writer = asyncio.StreamWriter(
        _MemoryTransport(reader, server_protocol),
        server_protocol,
        server_reader,
        routed_loop,
    )
    # Keep one conversation alive across samples, exercising the public
    # server pipeline and bounded recording buffers without a listener.
    task = routed_loop.create_task(
        routed_server.handle_http_connection(server_reader, server_writer)
    )
    try:
        yield reader, writer
    finally:
        writer.close()
        routed_loop.run_until_complete(writer.wait_closed())
        routed_loop.run_until_complete(task)
        routed_loop.run_until_complete(routed_server.aclose())


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


def test_server_roundtrip_routed_middleware(
    benchmark: BenchmarkFixture,
    routed_loop: asyncio.AbstractEventLoop,
    routed_server: AsyncHTTPTestServer,
    routed_connection: Connection,
) -> None:
    reader, writer = routed_connection

    def exchange_once() -> tuple[bytes, bytes]:
        return routed_loop.run_until_complete(exchange(reader, writer))

    # Warm the persistent connection outside measurement, also ensuring
    # that another sample can reuse it when benchmarks run as tests.
    exchange_once()
    head, body = benchmark(exchange_once)

    assert head.startswith(b"HTTP/1.1 200 OK\r\n")
    assert body == RESPONSE_BODY
    assert routed_server.last_request is not None
    assert routed_server.last_request.wire_raw_bytes == REQUEST
    assert routed_server.last_response is not None
    assert routed_server.last_response.wire_raw_bytes == head + body


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
