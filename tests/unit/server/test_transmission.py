from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, create_autospec

import pytest

from localstub.server import (
    ByteFlip,
    Delay,
    DropConnection,
    FaultyTransmission,
    ImmediateTransmission,
    RecordingStreamWriter,
    ThrottledTransmission,
    TruncateBody,
)


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
async def test_faulty_transmission_drop_connection_returns_abort() -> None:
    strategy = FaultyTransmission(faults=[DropConnection(after_bytes=2)])
    body = b"abcdef"

    writer = AsyncMock(spec=asyncio.StreamWriter)
    recorder = RecordingStreamWriter(writer)

    abort = await strategy.write_body(recorder, body)

    assert writer.write.call_args[0][0] == b"ab"
    assert abort is not None
    assert not abort.reset
    writer.close.assert_not_called()
    assert recorder.bytes_sent == b"ab"


@pytest.mark.asyncio
async def test_faulty_transmission_drop_connection_reset_propagates() -> None:
    strategy = FaultyTransmission(
        faults=[DropConnection(after_bytes=0, reset=True)]
    )

    writer = AsyncMock(spec=asyncio.StreamWriter)

    abort = await strategy.write_body(writer, b"abcdef")

    assert abort is not None
    assert abort.reset
    writer.write.assert_not_called()


@pytest.mark.asyncio
async def test_faulty_transmission_first_drop_wins() -> None:
    strategy = FaultyTransmission(
        faults=[
            DropConnection(after_bytes=1, reset=True),
            DropConnection(after_bytes=3),
        ]
    )

    writer = AsyncMock(spec=asyncio.StreamWriter)

    abort = await strategy.write_body(writer, b"abcdef")

    assert writer.write.call_args[0][0] == b"a"
    assert abort is not None
    assert abort.reset


@pytest.mark.asyncio
async def test_faulty_transmission_uses_injected_sleep() -> None:
    sleep = create_autospec(asyncio.sleep)
    strategy = FaultyTransmission(faults=[Delay(0.5)], sleep=sleep)

    writer = AsyncMock(spec=asyncio.StreamWriter)

    assert await strategy.write_body(writer, b"abc") is None
    sleep.assert_awaited_once_with(0.5)


@pytest.mark.asyncio
async def test_immediate_transmission_returns_none() -> None:
    writer = AsyncMock(spec=asyncio.StreamWriter)

    assert await ImmediateTransmission().write_body(writer, b"abc") is None


@pytest.mark.asyncio
async def test_throttled_transmission_returns_none() -> None:
    sleep = create_autospec(asyncio.sleep)
    strategy = ThrottledTransmission(chunk_size=1, delay=0, sleep=sleep)
    writer = AsyncMock(spec=asyncio.StreamWriter)

    assert await strategy.write_body(writer, b"abc") is None


def test_drop_connection_rejects_negative_after_bytes() -> None:
    with pytest.raises(ValueError, match="after_bytes"):
        DropConnection(after_bytes=-1)


def test_delay_rejects_negative_seconds() -> None:
    with pytest.raises(ValueError, match="seconds must be non-negative"):
        Delay(-0.1)


def test_truncate_body_rejects_negative_keep_bytes() -> None:
    with pytest.raises(ValueError, match="keep_bytes must be non-negative"):
        TruncateBody(-1)


def test_byte_flip_rejects_negative_offset() -> None:
    with pytest.raises(ValueError, match="offset must be non-negative"):
        ByteFlip(offset=-1)


@pytest.mark.parametrize("mask", [-1, 256])
def test_byte_flip_rejects_mask_outside_byte_range(mask: int) -> None:
    with pytest.raises(ValueError, match="mask must be between 0 and 255"):
        ByteFlip(offset=0, mask=mask)


def test_byte_flip_beyond_body_length_leaves_body_unchanged() -> None:
    result = ByteFlip(offset=3).apply(b"abc")

    assert result.body == b"abc"
