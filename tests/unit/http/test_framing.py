from __future__ import annotations

import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.http.framing import (
    ChunkScanError,
    chunk_payloads,
    content_length,
    is_chunked_transfer,
    scan_chunk_payloads,
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
    significant = digits.lstrip(b"0")
    if digits.isdigit() and len(significant) <= sys.get_int_max_str_digits():
        assert result == int(significant or b"0")
    else:
        assert result is None


def test_content_length_with_over_limit_digits_returns_none() -> None:
    value = b"1" * (sys.get_int_max_str_digits() + 1)

    assert content_length([(b"Content-Length", value)]) is None


def test_content_length_with_over_limit_leading_zeros_returns_value() -> None:
    value = b"0" * sys.get_int_max_str_digits() + b"42"

    assert content_length([(b"Content-Length", value)]) == 42


def test_content_length_with_only_zeros_returns_zero() -> None:
    assert content_length([(b"Content-Length", b"000")]) == 0


def test_scan_chunked_body_resumes_after_complete_chunks() -> None:
    buffer = bytearray(b"3\r\none\r\n")

    first_scan = scan_chunked_body(buffer)

    assert first_scan.end is None
    assert first_scan.resume_from == len(buffer)

    buffer.extend(b"3\r\ntwo\r\n0\r\n\r\n")
    second_scan = scan_chunked_body(buffer, first_scan.resume_from)

    assert second_scan.end == len(buffer)


def test_is_chunked_transfer_with_chunked_token_returns_true() -> None:
    assert is_chunked_transfer([(b"Transfer-Encoding", b"gzip, chunked")])


def test_is_chunked_transfer_with_other_encodings_only_returns_false() -> None:
    assert not is_chunked_transfer([(b"Transfer-Encoding", b"gzip")])


def test_is_chunked_transfer_finds_chunked_in_later_header() -> None:
    headers = [
        (b"Transfer-Encoding", b"gzip"),
        (b"transfer-encoding", b"chunked"),
    ]

    assert is_chunked_transfer(headers)


def test_is_chunked_transfer_without_transfer_encoding_returns_false() -> None:
    assert not is_chunked_transfer([(b"Content-Length", b"5")])


def test_scan_chunk_payloads_empty_chunk_size_raises_error_at_first_byte() -> (
    None
):
    with pytest.raises(ChunkScanError) as excinfo:
        scan_chunk_payloads(bytearray(b"\r\nhello\r\n"))

    assert excinfo.value.offset == 1


def test_scan_chunk_payloads_bare_cr_after_size_raises_error() -> None:
    with pytest.raises(ChunkScanError) as excinfo:
        scan_chunk_payloads(bytearray(b"5\rXhello\r\n"))

    assert excinfo.value.offset == 2


def test_scan_chunk_payloads_extension_without_size_raises_error() -> None:
    with pytest.raises(ChunkScanError) as excinfo:
        scan_chunk_payloads(bytearray(b";ext=1\r\nhello\r\n"))

    assert excinfo.value.offset == 1


def test_scan_chunk_payloads_reports_complete_chunk_data_ranges() -> None:
    buffer = bytearray(b"5\r\nhello\r\n3\r\nabc\r\n")

    scan = scan_chunk_payloads(buffer)

    assert scan.payloads == ((3, 8), (13, 16))
    assert scan.partial is None
    assert scan.end is None
    assert scan.resume_from == len(buffer)


def test_scan_chunk_payloads_reports_partial_chunk_data() -> None:
    buffer = bytearray(b"5\r\nhello\r\n3\r\nab")

    scan = scan_chunk_payloads(buffer)

    assert scan.payloads == ((3, 8),)
    assert scan.partial == (13, 15)
    assert scan.end is None
    assert scan.resume_from == 10


def test_scan_chunk_payloads_size_line_cut_after_cr_is_incomplete() -> None:
    scan = scan_chunk_payloads(bytearray(b"5\r\nhello\r\n5\r"))

    assert scan.payloads == ((3, 8),)
    assert scan.partial is None
    assert scan.end is None
    assert scan.resume_from == 10


@pytest.mark.parametrize(
    ("buffer", "payloads", "partial", "resume_from"),
    [
        (b"5\r\n", (), (3, 3), 0),
        (b"5\r\nhello\r\n3;ext=1\r\n", ((3, 8),), (19, 19), 10),
    ],
)
def test_scan_chunk_payloads_chunk_without_data_has_empty_partial(
    buffer: bytes,
    payloads: tuple[tuple[int, int], ...],
    partial: tuple[int, int],
    resume_from: int,
) -> None:
    scan = scan_chunk_payloads(bytearray(buffer))

    assert scan.payloads == payloads
    assert scan.partial == partial
    assert scan.end is None
    assert scan.resume_from == resume_from


def test_scan_chunk_payloads_terminal_chunk_with_trailers_sets_end() -> None:
    buffer = bytearray(b"5\r\nhello\r\n0\r\nX-Checksum: abc\r\n\r\n")

    scan = scan_chunk_payloads(buffer)

    assert scan.payloads == ((3, 8),)
    assert scan.end == len(buffer)
    assert scan.resume_from == len(buffer)


def test_scan_chunk_payloads_incomplete_trailers_resume_at_terminal() -> None:
    buffer = bytearray(b"5\r\nhello\r\n0\r\nX-Checksum: abc")

    scan = scan_chunk_payloads(buffer)

    assert scan.end is None
    assert scan.resume_from == 10


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
