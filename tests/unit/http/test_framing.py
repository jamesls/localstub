import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.http.framing import (
    ChunkScanError,
    chunk_payloads,
    content_length,
    scan_chunked_body,
)

from .strategies import (
    ChunkedBody,
    chunked_bodies,
    fragments_of,
)

CONTENT_LENGTH_VALUES = st.one_of(
    st.binary(),
    st.text(alphabet="0123456789+-_ \t", min_size=1).map(
        lambda text: text.encode("ascii")
    ),
)


@given(value=CONTENT_LENGTH_VALUES)
def test_content_length_with_arbitrary_value_accepts_only_digits(
    value: bytes,
) -> None:
    result = content_length([(b"Content-Length", value)])

    digits = value.strip()
    if digits.isdigit():
        assert result == int(digits)
    else:
        assert result is None


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


@st.composite
def _incremental_scan_cases(
    draw: st.DrawFn,
) -> tuple[ChunkedBody, list[bytes]]:
    body = draw(chunked_bodies())
    return body, draw(fragments_of(body.encoded))


@st.composite
def _corrupted_chunked_encodings(draw: st.DrawFn) -> bytes:
    encoded = bytearray(draw(chunked_bodies()).encoded)
    index = draw(st.integers(min_value=0, max_value=len(encoded) - 1))
    encoded[index] = draw(st.integers(min_value=0, max_value=255))
    return bytes(encoded)


SCANNER_INPUTS = st.one_of(
    st.binary(max_size=128),
    _corrupted_chunked_encodings(),
)


@given(body=chunked_bodies())
def test_chunk_payloads_roundtrip_recovers_encoded_payloads(
    body: ChunkedBody,
) -> None:
    assert chunk_payloads(bytearray(body.encoded)) == body.payloads


@given(body=chunked_bodies())
def test_scan_chunked_body_end_matches_encoded_length(
    body: ChunkedBody,
) -> None:
    scan = scan_chunked_body(bytearray(body.encoded))

    assert scan.end == len(body.encoded)
    assert scan.resume_from == scan.end


@given(case=_incremental_scan_cases())
def test_scan_chunked_body_incremental_delivery_matches_one_shot(
    case: tuple[ChunkedBody, list[bytes]],
) -> None:
    body, fragments = case
    buffer = bytearray()
    resume = 0

    for index, fragment in enumerate(fragments):
        buffer.extend(fragment)
        scan = scan_chunked_body(buffer, resume)

        assert resume <= scan.resume_from <= len(buffer)
        resume = scan.resume_from
        if index < len(fragments) - 1:
            assert scan.end is None
        else:
            assert scan.end == len(body.encoded)


@given(body=chunked_bodies(), garbage=st.binary(min_size=1, max_size=32))
def test_scan_chunked_body_ignores_bytes_after_terminal_chunk(
    body: ChunkedBody,
    garbage: bytes,
) -> None:
    scan = scan_chunked_body(bytearray(body.encoded + garbage))

    assert scan.end == len(body.encoded)


@given(body=chunked_bodies(), garbage=st.binary(min_size=1, max_size=32))
def test_chunk_payloads_ignores_bytes_after_terminal_chunk(
    body: ChunkedBody,
    garbage: bytes,
) -> None:
    assert chunk_payloads(bytearray(body.encoded + garbage)) == body.payloads


@given(data=SCANNER_INPUTS)
def test_scan_chunked_body_arbitrary_input_scans_or_raises_scan_error(
    data: bytes,
) -> None:
    buffer = bytearray(data)
    try:
        scan = scan_chunked_body(buffer)
    except ChunkScanError as exc:
        assert 1 <= exc.offset <= len(buffer)
    else:
        assert 0 <= scan.resume_from <= len(buffer)
        if scan.end is not None:
            assert scan.end == scan.resume_from


@given(data=SCANNER_INPUTS)
def test_chunk_payloads_arbitrary_input_returns_list_or_raises_scan_error(
    data: bytes,
) -> None:
    buffer = bytearray(data)
    try:
        payloads = chunk_payloads(buffer)
    except ChunkScanError as exc:
        assert 1 <= exc.offset <= len(buffer)
    else:
        assert all(isinstance(payload, bytes) for payload in payloads)
