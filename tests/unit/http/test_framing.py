import pytest

from localstub.http.framing import (
    ChunkScanError,
    chunk_payloads,
    scan_chunked_body,
)


def test_scan_chunked_body_resumes_after_complete_chunks() -> None:
    buffer = bytearray(b"3\r\none\r\n")

    first_scan = scan_chunked_body(buffer)

    assert first_scan.end is None
    assert first_scan.resume_from == len(buffer)

    buffer.extend(b"3\r\ntwo\r\n0\r\n\r\n")
    second_scan = scan_chunked_body(buffer, first_scan.resume_from)

    assert second_scan.end == len(buffer)


def test_chunk_payloads_returns_single_chunk_data() -> None:
    buffer = bytearray(b"5\r\nhello\r\n0\r\n\r\n")

    assert chunk_payloads(buffer) == [b"hello"]


def test_chunk_payloads_returns_multiple_chunk_data() -> None:
    buffer = bytearray(b"5\r\nHello\r\n6\r\n World\r\n0\r\n\r\n")

    assert chunk_payloads(buffer) == [b"Hello", b" World"]


def test_chunk_payloads_skips_chunk_extensions() -> None:
    buffer = bytearray(b"5;name=value\r\nhello\r\n0\r\n\r\n")

    assert chunk_payloads(buffer) == [b"hello"]


def test_chunk_payloads_stops_at_terminal_chunk() -> None:
    buffer = bytearray(b"5\r\nhello\r\n0\r\n\r\nGET /next HTTP/1.1\r\n\r\n")

    assert chunk_payloads(buffer) == [b"hello"]


def test_chunk_payloads_ignores_trailers() -> None:
    buffer = bytearray(b"5\r\nhello\r\n0\r\nX-Checksum: abc\r\n\r\n")

    assert chunk_payloads(buffer) == [b"hello"]


def test_chunk_payloads_truncated_returns_complete_chunks() -> None:
    buffer = bytearray(b"5\r\nhello\r\n3\r\nab")

    assert chunk_payloads(buffer) == [b"hello"]


def test_chunk_payloads_empty_buffer_returns_no_payloads() -> None:
    assert chunk_payloads(bytearray()) == []


def test_chunk_payloads_malformed_size_raises_error() -> None:
    buffer = bytearray(b"XYZ\r\nhello\r\n")

    with pytest.raises(ChunkScanError):
        chunk_payloads(buffer)


def test_chunk_payloads_bad_chunk_terminator_raises_error() -> None:
    buffer = bytearray(b"5\r\nhelloXY0\r\n\r\n")

    with pytest.raises(ChunkScanError):
        chunk_payloads(buffer)
