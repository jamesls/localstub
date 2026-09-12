from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, create_autospec

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.recording import BoundedByteBuffer
from localstub.server import RecordingStreamWriter


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


def test_recording_writer_write_eof_delegates_to_wrapped_writer() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    recorder = RecordingStreamWriter(writer)

    recorder.write_eof()

    writer.write_eof.assert_called_once_with()


@pytest.mark.asyncio
async def test_recording_writer_wait_closed_awaits_wrapped_writer() -> None:
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    recorder = RecordingStreamWriter(writer)

    await recorder.wait_closed()

    writer.wait_closed.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_recording_writer_cancelled_wait_keeps_shared_waiter() -> None:
    shared = asyncio.get_running_loop().create_future()

    async def wait_closed() -> None:
        await shared

    writer = create_autospec(asyncio.StreamWriter, instance=True)
    writer.wait_closed.side_effect = wait_closed
    recorder = RecordingStreamWriter(writer)
    waiting = asyncio.create_task(recorder.wait_closed())
    await asyncio.sleep(0)

    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting

    assert not shared.cancelled()
    shared.set_result(None)
    await asyncio.sleep(0)
