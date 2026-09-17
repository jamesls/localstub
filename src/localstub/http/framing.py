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


@dataclass(frozen=True)
class ChunkPayloadScan:
    """Chunk payload locations found by scanning chunk framing.

    ``payloads`` holds the ``(start, end)`` data ranges of the complete
    chunks after the scan start, in wire order.  ``partial`` is the
    data range buffered so far for the first incomplete chunk, once
    its size line is complete; the range is empty until any of its
    data arrives.  ``end`` is the offset past the chunked body once
    the terminal chunk and any trailers are complete.  ``resume_from``
    is the chunk-size boundary of the first incomplete chunk, so a
    caller can rescan from there after appending data.
    """

    payloads: tuple[tuple[int, int], ...]
    partial: tuple[int, int] | None
    end: int | None
    resume_from: int


def content_length(headers: list[tuple[bytes, bytes]]) -> int | None:
    for name, value in reversed(headers):
        if name.lower() != b"content-length":
            continue
        # RFC 9110: Content-Length is 1*DIGIT.  int() alone is too
        # lenient (accepts sign prefixes and underscores).
        digits = value.strip()
        if not digits.isdigit():
            return None
        # Leading zeros are valid digits that leave the value unchanged
        # but count toward the interpreter's conversion limit.
        digits = digits.lstrip(b"0") or b"0"
        try:
            return int(digits)
        except ValueError:
            # int() refuses digit strings longer than the
            # interpreter's conversion limit
            # (sys.get_int_max_str_digits(), 4300 by default).
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


def chunk_payloads(buffer: bytearray) -> list[bytes]:
    """Extract the chunk data from chunked transfer framing.

    Returns the payload of each complete chunk before the terminal
    zero-size chunk, skipping chunk extensions and trailers.  Stops
    at the terminal chunk or the first incomplete chunk, so bytes
    that follow the chunked body are never touched.

    Raises ChunkScanError where scan_chunked_body would: on a
    malformed chunk-size line or a bad chunk-data terminator.
    """
    payloads: list[bytes] = []
    index = 0

    while True:
        chunk_size_line = _scan_chunk_size_line(buffer, index)
        if chunk_size_line is None:
            return payloads

        chunk_size, chunk_data_start = chunk_size_line
        if chunk_size == 0:
            return payloads

        chunk_data_end = chunk_data_start + chunk_size
        if chunk_data_end + 2 > len(buffer):
            return payloads
        if not buffer.startswith(CRLF, chunk_data_end):
            raise ChunkScanError(chunk_data_end + 1)

        payloads.append(bytes(buffer[chunk_data_start:chunk_data_end]))
        index = chunk_data_end + 2


def scan_chunked_body(
    buffer: bytearray,
    start: int = 0,
) -> ChunkScanResult:
    """Scan chunk framing from a known chunk-size boundary.

    `resume_from` identifies the first incomplete chunk so callers can
    continue scanning after appending data without revisiting complete chunks.
    """
    scan = scan_chunk_payloads(buffer, start)
    return ChunkScanResult(end=scan.end, resume_from=scan.resume_from)


def _terminal_chunk_end(buffer: bytearray, data_start: int) -> int | None:
    """Return the offset past the chunked body once its end is complete."""
    if buffer.startswith(CRLF, data_start):
        return data_start + 2
    trailer_end = buffer.find(HEADER_TERMINATOR, data_start)
    if trailer_end == -1:
        return None
    return trailer_end + len(HEADER_TERMINATOR)


def _check_chunk_terminator(buffer: bytearray, data_end: int) -> None:
    """Raise ChunkScanError at the first byte that is not the chunk CRLF."""
    for offset, expected in enumerate(CRLF, start=data_end):
        if buffer[offset] != expected:
            raise ChunkScanError(offset + 1)


def scan_chunk_payloads(
    buffer: bytearray,
    start: int = 0,
) -> ChunkPayloadScan:
    """Locate chunk payloads from a known chunk-size boundary.

    Complete chunks are reported with their data ranges so a caller
    can map a payload byte count to a wire offset; the first
    incomplete chunk is reported as ``partial`` once its size line is
    complete.  Raises ChunkScanError on a malformed chunk-size line or
    a bad chunk-data terminator.
    """
    payloads: list[tuple[int, int]] = []
    index = start

    while True:
        chunk_size_line = _scan_chunk_size_line(buffer, index)
        if chunk_size_line is None:
            return ChunkPayloadScan(tuple(payloads), None, None, index)

        chunk_size, data_start = chunk_size_line

        if chunk_size == 0:
            end = _terminal_chunk_end(buffer, data_start)
            resume_from = index if end is None else end
            return ChunkPayloadScan(tuple(payloads), None, end, resume_from)

        data_end = data_start + chunk_size
        if data_end + 2 > len(buffer):
            partial = (data_start, min(data_end, len(buffer)))
            return ChunkPayloadScan(tuple(payloads), partial, None, index)
        _check_chunk_terminator(buffer, data_end)

        payloads.append((data_start, data_end))
        index = data_end + 2
