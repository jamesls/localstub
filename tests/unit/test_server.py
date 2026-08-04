import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from localstub.server import (
    AsyncHTTPTestServer,
    ByteFlip,
    Delay,
    DropConnection,
    FaultyTransmission,
    HTTPRequest,
    HTTPResponse,
    ImmediateTransmission,
    RecordingStreamWriter,
    ThrottledTransmission,
    TruncateBody,
)


def test_server_url_raises_when_not_started() -> None:
    server = AsyncHTTPTestServer()
    with pytest.raises(RuntimeError, match="Server not started yet"):
        _ = server.url


def test_server_handler_getter() -> None:
    def handler(req: HTTPRequest) -> HTTPResponse:
        _ = req
        return HTTPResponse.json({"test": "value"})

    server = AsyncHTTPTestServer(handler=handler)
    assert server.handler is handler


@pytest.mark.asyncio
async def test_immediate_transmission_sends_all_at_once() -> None:
    strategy = ImmediateTransmission()
    body = b"x" * 10000

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    await strategy.write_body(recorder, body)

    assert writer.write.call_count == 1
    assert writer.write.call_args[0][0] == body
    assert writer.drain.call_count == 1
    assert recorder.bytes_sent == body


@pytest.mark.asyncio
async def test_immediate_transmission_without_tracking() -> None:
    strategy = ImmediateTransmission()
    body = b"test data"

    writer = AsyncMock(spec=asyncio.StreamWriter)

    await strategy.write_body(writer, body)

    assert writer.write.call_count == 1
    assert writer.write.call_args[0][0] == body
    assert writer.drain.call_count == 1


@pytest.mark.asyncio
async def test_throttled_transmission_chunks_body() -> None:
    strategy = ThrottledTransmission(chunk_size=100, delay=0.01)
    body = b"x" * 250

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    start = time.time()
    await strategy.write_body(recorder, body)
    elapsed = time.time() - start

    assert writer.write.call_count == 3
    assert writer.drain.call_count == 3

    chunks = [call[0][0] for call in writer.write.call_args_list]
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 50

    assert recorder.bytes_sent == body
    assert elapsed >= 0.018


@pytest.mark.asyncio
async def test_throttled_transmission_single_chunk() -> None:
    strategy = ThrottledTransmission(chunk_size=1000, delay=0.01)
    body = b"small"

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    start = time.time()
    await strategy.write_body(recorder, body)
    elapsed = time.time() - start

    assert writer.write.call_count == 1
    assert writer.drain.call_count == 1
    assert recorder.bytes_sent == body
    assert elapsed < 0.005


@pytest.mark.asyncio
async def test_throttled_transmission_exact_multiple() -> None:
    strategy = ThrottledTransmission(chunk_size=100, delay=0.01)
    body = b"x" * 200

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    start = time.time()
    await strategy.write_body(recorder, body)
    elapsed = time.time() - start

    assert writer.write.call_count == 2
    assert writer.drain.call_count == 2

    chunks = [call[0][0] for call in writer.write.call_args_list]
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert recorder.bytes_sent == body

    assert elapsed >= 0.008


def test_throttled_transmission_rejects_zero_chunk_size() -> None:
    with pytest.raises(
        ValueError, match="chunk_size must be a positive integer"
    ):
        ThrottledTransmission(chunk_size=0, delay=0.01)


def test_throttled_transmission_rejects_negative_chunk_size() -> None:
    with pytest.raises(
        ValueError, match="chunk_size must be a positive integer"
    ):
        ThrottledTransmission(chunk_size=-5, delay=0.01)


@pytest.mark.asyncio
async def test_faulty_transmission_delay() -> None:
    strategy = FaultyTransmission(faults=[Delay(0.05)])
    body = b"abc"

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    start = time.time()
    await strategy.write_body(recorder, body)
    elapsed = time.time() - start

    assert elapsed >= 0.045
    assert recorder.bytes_sent == body


@pytest.mark.asyncio
async def test_faulty_transmission_truncate() -> None:
    strategy = FaultyTransmission(faults=[TruncateBody(3)])
    body = b"hello"

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    await strategy.write_body(recorder, body)

    assert writer.write.call_args[0][0] == b"hel"
    assert recorder.bytes_sent == b"hel"


@pytest.mark.asyncio
async def test_faulty_transmission_byte_flip() -> None:
    strategy = FaultyTransmission(faults=[ByteFlip(offset=1, mask=0x0F)])
    body = b"\x00\x01\x02"

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    await strategy.write_body(recorder, body)

    expected = b"\x00\x0e\x02"
    assert writer.write.call_args[0][0] == expected
    assert recorder.bytes_sent == expected


@pytest.mark.asyncio
async def test_faulty_transmission_drop_connection() -> None:
    strategy = FaultyTransmission(faults=[DropConnection(after_bytes=2)])
    body = b"abcdef"

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    await strategy.write_body(recorder, body)

    assert writer.write.call_args[0][0] == b"ab"
    assert writer.close.call_count == 1
    assert writer.wait_closed.call_count == 1
    assert recorder.bytes_sent == b"ab"


def test_zero_recording_buffer_size_raises_value_error():
    with pytest.raises(ValueError, match="at least 1"):
        AsyncHTTPTestServer(recording_buffer_size=0)


def test_negative_recording_buffer_size_raises_value_error():
    with pytest.raises(ValueError, match="at least 1"):
        AsyncHTTPTestServer(recording_buffer_size=-1)
