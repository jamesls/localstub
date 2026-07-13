from __future__ import annotations

from dataclasses import dataclass

HEADER_TERMINATOR = b"\r\n\r\n"
CRLF = b"\r\n"


class ChunkScanError(Exception):
    def __init__(self, offset: int) -> None:
        self.offset = offset


@dataclass(frozen=True)
class ChunkScanResult:
    end: int | None
    resume_from: int


def content_length(headers: list[tuple[bytes, bytes]]) -> int | None:
    for name, value in reversed(headers):
        if name.lower() != b"content-length":
            continue
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def is_chunked_transfer(headers: list[tuple[bytes, bytes]]) -> bool:
    for name, value in headers:
        if name.lower() != b"transfer-encoding":
            continue
        for token in value.split(b","):
            if token.strip().lower() == b"chunked":
                return True
    return False


def _is_hex_digit(value: int) -> bool:
    return (
        ord("0") <= value <= ord("9")
        or ord("A") <= value <= ord("F")
        or ord("a") <= value <= ord("f")
    )


def _parse_chunk_size(
    buffer: bytearray,
    start: int,
    end: int,
) -> int:
    if start == end:
        raise ChunkScanError(start + 1)
    return int(bytes(buffer[start:end]), 16)


def _scan_chunk_size_line(
    buffer: bytearray,
    start: int,
) -> tuple[int, int] | None:
    size_end = start

    while True:
        if size_end >= len(buffer):
            return None

        current = buffer[size_end]
        if _is_hex_digit(current):
            size_end += 1
            continue

        if current == ord(";"):
            line_end = buffer.find(CRLF, size_end)
            if line_end == -1:
                return None
            return _parse_chunk_size(buffer, start, size_end), line_end + 2

        if current == ord("\r"):
            if size_end + 1 >= len(buffer):
                return None
            if buffer[size_end + 1] != ord("\n"):
                raise ChunkScanError(size_end + 1)
            return _parse_chunk_size(buffer, start, size_end), size_end + 2

        raise ChunkScanError(size_end + 1)


def scan_chunked_body(
    buffer: bytearray,
    start: int = 0,
) -> ChunkScanResult:
    """Scan chunk framing from a known chunk-size boundary.

    `resume_from` identifies the first incomplete chunk so callers can
    continue scanning after appending data without revisiting complete chunks.
    """
    index = start

    while True:
        chunk_size_line = _scan_chunk_size_line(buffer, index)
        if chunk_size_line is None:
            return ChunkScanResult(end=None, resume_from=index)

        chunk_size, chunk_data_start = chunk_size_line

        if chunk_size == 0:
            if buffer[chunk_data_start : chunk_data_start + 2] == CRLF:
                end = chunk_data_start + 2
                return ChunkScanResult(end=end, resume_from=end)
            trailer_end = buffer.find(HEADER_TERMINATOR, chunk_data_start)
            if trailer_end == -1:
                return ChunkScanResult(end=None, resume_from=index)
            end = trailer_end + len(HEADER_TERMINATOR)
            return ChunkScanResult(end=end, resume_from=end)

        chunk_data_end = chunk_data_start + chunk_size
        if chunk_data_end + 2 > len(buffer):
            return ChunkScanResult(end=None, resume_from=index)
        if buffer[chunk_data_end : chunk_data_end + 2] != CRLF:
            mismatch = chunk_data_end
            while mismatch < len(buffer) and mismatch < chunk_data_end + 2:
                expected = (
                    ord("\r") if mismatch == chunk_data_end else ord("\n")
                )
                if buffer[mismatch] != expected:
                    raise ChunkScanError(mismatch + 1)
                mismatch += 1
            return ChunkScanResult(end=None, resume_from=index)

        index = chunk_data_end + 2
