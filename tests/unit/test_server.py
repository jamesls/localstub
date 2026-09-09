import asyncio
import time
from unittest.mock import AsyncMock, create_autospec

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.http.clients.asyncio import AsyncioClient
from localstub.middleware import ResponderContext
from localstub.recording import BoundedByteBuffer, TrafficRecorder
from localstub.server import (
    AsyncHTTPTestServer,
    ByteFlip,
    Delay,
    DropConnection,
    FaultyTransmission,
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


def test_forward_proxy_with_injected_client_raises_value_error() -> None:
    with pytest.raises(ValueError, match="not both"):
        AsyncHTTPTestServer(
            forward_proxy=True, upstream_client=AsyncioClient()
        )


def test_server_handler_getter() -> None:
    def handler(ctx: ResponderContext) -> HTTPResponse:
        _ = ctx
        return HTTPResponse.json({"test": "value"})

    server = AsyncHTTPTestServer(handler=handler)
    assert server.handler is handler


def test_server_accepts_recorder_in_legacy_positional_slot() -> None:
    recorder = TrafficRecorder(None)

    server = AsyncHTTPTestServer(
        "127.0.0.1",
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        recorder,
    )

    assert server.requests == []


def test_recording_writer_holds_only_current_response() -> None:
    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    recorder.start_response()
    recorder.write(b"first response")
    recorder.start_response()
    recorder.write(b"second response")

    assert recorder.bytes_sent == b"second response"


def test_recording_writer_reuses_single_write_through_empty_writes() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    recorder = RecordingStreamWriter(writer)
    body = b"x" * 10000

    assert recorder.bytes_sent == b""
    recorder.write(b"")
    recorder.write(body)
    recorder.write(b"")

    assert recorder.bytes_sent is body
    assert [call.args[0] for call in writer.write.call_args_list] == [
        b"",
        body,
        b"",
    ]


def test_recording_writer_coalesces_small_writes_around_large_write() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    recorder = RecordingStreamWriter(writer, coalesce_size=4)

    recorder.writelines([b"ab", b"c", b"defgh", b"i", b"jkl", b"m"])

    assert recorder.bytes_sent == b"abcdefghijklm"


def test_recording_writer_coalesce_size_below_one_raises_value_error() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)

    with pytest.raises(ValueError, match="coalesce_size"):
        RecordingStreamWriter(writer, coalesce_size=0)


@given(
    coalesce_size=st.integers(min_value=1, max_value=16),
    chunks=st.lists(st.binary(max_size=64), max_size=40),
)
def test_recording_writer_bytes_sent_matches_every_write(
    coalesce_size: int, chunks: list[bytes]
) -> None:
    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer, coalesce_size=coalesce_size)

    expected = b""
    for chunk in chunks:
        recorder.write(chunk)
        expected += chunk
        assert recorder.bytes_sent == expected

    assert [call.args[0] for call in writer.write.call_args_list] == chunks


def test_recording_writer_snapshots_survive_more_writes_and_reset() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    sent = BoundedByteBuffer(5)
    recorder = RecordingStreamWriter(writer, sent)
    recorder.write(b"ab")
    first = recorder.bytes_sent
    recorder.write(b"cd")
    second = recorder.bytes_sent
    recorder.writelines([b"", b"ef", b"gh"])
    third = recorder.bytes_sent
    recorder.start_response()

    assert recorder.bytes_sent == b""
    recorder.write(b"ij")
    assert first == b"ab"
    assert second == b"abcd"
    assert third == b"abcdefgh"
    assert recorder.bytes_sent == b"ij"
    assert bytes(sent) == b"fghij"
    assert sent.dropped == 5
    assert [call.args[0] for call in writer.write.call_args_list] == [
        b"ab",
        b"cd",
        b"",
        b"ef",
        b"gh",
        b"ij",
    ]


@pytest.mark.parametrize("prefix", [b"", b"previous"])
def test_recording_writer_preserves_capture_on_write_failure(
    prefix: bytes,
) -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    sent = BoundedByteBuffer(100)
    recorder = RecordingStreamWriter(writer, sent)
    recorder.write(prefix)
    writer.write.side_effect = ConnectionError("closed")

    with pytest.raises(ConnectionError, match="closed"):
        recorder.write(b"failed")

    assert recorder.bytes_sent == prefix + b"failed"
    assert bytes(sent) == prefix + b"failed"


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
    sleep = create_autospec(asyncio.sleep)
    strategy = ThrottledTransmission(chunk_size=100, delay=0.01, sleep=sleep)
    body = b"x" * 250

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    await strategy.write_body(recorder, body)

    assert writer.write.call_count == 3
    assert writer.drain.call_count == 3

    chunks = [call[0][0] for call in writer.write.call_args_list]
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 50

    assert recorder.bytes_sent == body
    assert sleep.await_count == 2
    sleep.assert_awaited_with(0.01)


@pytest.mark.asyncio
async def test_throttled_transmission_single_chunk() -> None:
    sleep = create_autospec(asyncio.sleep)
    strategy = ThrottledTransmission(chunk_size=1000, delay=0.01, sleep=sleep)
    body = b"small"

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    await strategy.write_body(recorder, body)

    assert writer.write.call_count == 1
    assert writer.drain.call_count == 1
    assert recorder.bytes_sent == body
    sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_throttled_transmission_one_byte_chunks_records_body() -> None:
    sleep = create_autospec(asyncio.sleep)
    strategy = ThrottledTransmission(chunk_size=1, delay=0, sleep=sleep)
    body = bytes(range(256)) * 17

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    await strategy.write_body(recorder, body)

    assert writer.write.call_count == len(body)
    assert recorder.bytes_sent == body
    assert sleep.await_count == len(body) - 1


@pytest.mark.asyncio
async def test_throttled_transmission_exact_multiple() -> None:
    sleep = create_autospec(asyncio.sleep)
    strategy = ThrottledTransmission(chunk_size=100, delay=0.01, sleep=sleep)
    body = b"x" * 200

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    await strategy.write_body(recorder, body)

    assert writer.write.call_count == 2
    assert writer.drain.call_count == 2

    chunks = [call[0][0] for call in writer.write.call_args_list]
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert recorder.bytes_sent == body

    assert sleep.await_count == 1
    sleep.assert_awaited_with(0.01)


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


def test_negative_max_connection_bytes_raises_value_error() -> None:
    with pytest.raises(ValueError, match="must be non-negative or None"):
        AsyncHTTPTestServer(max_connection_bytes=-1)
