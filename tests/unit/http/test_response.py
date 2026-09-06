from __future__ import annotations

import asyncio
import itertools
from unittest.mock import AsyncMock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.http.headers import Headers
from localstub.http.response import (
    AsyncMultiResponseParser,
    AsyncResponseParser,
    ParsedResponse,
    RecordedHTTPResponse,
    ResponseProtocol,
)
from localstub.http.responsespec import HTTPResponse

from .strategies import (
    HEADER_VALUE_ALPHABET,
    chunked_bodies,
    fragments_of,
    header_items,
)


def test_recorded_http_response_delegates_to_response_value() -> None:
    recorded = RecordedHTTPResponse(
        response=HTTPResponse(
            status=201,
            headers=Headers.from_items([
                ("Content-Type", "text/plain"),
                ("Set-Cookie", "a=1"),
                ("Set-Cookie", "b=2"),
            ]),
            body=b"created",
        ),
        reason="Created",
        wire_raw_bytes=b"HTTP/1.1 201 Created\r\n\r\ncreated",
    )

    assert recorded.status == 201
    assert recorded.headers["Content-Type"] == "text/plain"
    assert recorded.headers.get_all("Set-Cookie") == ["a=1", "b=2"]
    assert recorded.body == b"created"
    assert recorded.reason == "Created"
    assert recorded.wire_raw_bytes == b"HTTP/1.1 201 Created\r\n\r\ncreated"


def test_parsed_response_default_values() -> None:
    response = ParsedResponse()

    assert response.status_code is None
    assert response.status_text is None
    assert response.http_version is None
    assert response.headers == []
    assert response.body_parts == []
    assert not response.is_complete


def test_parsed_response_body_property_empty() -> None:
    response = ParsedResponse()
    assert response.body == b""


def test_parsed_response_body_property_single_part() -> None:
    response = ParsedResponse(body_parts=[b"response body"])
    assert response.body == b"response body"


def test_parsed_response_body_property_multiple_parts() -> None:
    response = ParsedResponse(body_parts=[b"part1", b"part2", b"part3"])
    assert response.body == b"part1part2part3"


def test_response_protocol_on_message_begin_resets_result() -> None:
    protocol = ResponseProtocol()
    protocol.result.status_code = 200
    protocol.result.status_text = b"OK"

    protocol.on_message_begin()

    assert protocol.result.status_code is None
    assert protocol.result.status_text is None


def test_response_protocol_on_status_sets_status_text() -> None:
    protocol = ResponseProtocol()
    protocol.on_status(b"OK")

    assert protocol.result.status_text == b"OK"


def test_response_protocol_on_header_adds_headers() -> None:
    protocol = ResponseProtocol()
    protocol.on_header(b"Content-Type", b"text/html")
    protocol.on_header(b"Content-Length", b"100")

    assert len(protocol.result.headers) == 2
    assert protocol.result.headers[0] == (b"Content-Type", b"text/html")
    assert protocol.result.headers[1] == (b"Content-Length", b"100")


def test_response_protocol_on_body_adds_body_parts() -> None:
    protocol = ResponseProtocol()
    protocol.on_body(b"response data")

    assert protocol.result.body_parts == [b"response data"]


def test_response_protocol_on_message_complete_sets_complete_flag() -> None:
    protocol = ResponseProtocol()

    assert not protocol.result.is_complete

    protocol.on_message_complete()

    assert protocol.result.is_complete


def test_response_protocol_chunk_callbacks_are_noops() -> None:
    protocol = ResponseProtocol()

    protocol.on_chunk_header()
    protocol.on_chunk_complete()


def test_response_protocol_preserves_completed_message() -> None:
    protocol = ResponseProtocol()

    protocol.on_message_begin()
    protocol.on_status(b"OK")
    protocol.on_header(b"Content-Type", b"text/html")
    protocol.on_body(b"first body")
    protocol.on_message_complete()

    assert protocol.result.is_complete
    assert protocol.result.status_text == b"OK"
    assert protocol.result.headers == [(b"Content-Type", b"text/html")]
    assert protocol.result.body_parts == [b"first body"]

    protocol.on_message_begin()
    protocol.on_status(b"Not Found")
    protocol.on_header(b"Content-Type", b"text/plain")
    protocol.on_body(b"second body")
    protocol.on_message_complete()

    assert protocol.result.status_text == b"OK"
    assert protocol.result.headers == [(b"Content-Type", b"text/html")]
    assert protocol.result.body_parts == [b"first body"]


def test_response_protocol_on_status_accumulates_chunks() -> None:
    protocol = ResponseProtocol()

    protocol.on_status(b"Not")
    protocol.on_status(b" ")
    protocol.on_status(b"Found")

    assert protocol.result.status_text == b"Not Found"


def test_response_protocol_on_headers_complete_without_parser_is_noop() -> (
    None
):
    protocol = ResponseProtocol()
    protocol.result.status_text = b"OK"

    protocol.on_headers_complete()

    assert protocol.result.status_text == b"OK"
    assert protocol.result.status_code is None
    assert protocol.result.http_version is None


@pytest.mark.asyncio
async def test_async_response_parser_parse_simple_response() -> None:
    response_data = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/plain\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"Hello"
    )
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.status_code == 200
    assert parsed.status_text == b"OK"
    assert parsed.http_version == "1.1"
    assert parsed.body == b"Hello"
    assert wire_bytes == response_data


@pytest.mark.asyncio
async def test_async_response_parser_parse_chunked_response() -> None:
    response_data = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nHello\r\n"
        b"0\r\n\r\n"
    )
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is not None
    assert parsed.status_code == 200
    assert parsed.body == b"Hello"


@pytest.mark.asyncio
async def test_async_response_parser_parse_close_delimited_response() -> None:
    response_data = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/plain\r\n"
        b"\r\n"
        b"Response body without length"
    )
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is not None
    assert parsed.status_code == 200
    assert parsed.body == b"Response body without length"


@pytest.mark.asyncio
async def test_async_response_parser_parse_empty_response_body() -> None:
    response_data = b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n"
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is not None
    assert parsed.status_code == 204
    assert parsed.body == b""


@pytest.mark.asyncio
async def test_async_response_parser_parse_eof_before_headers() -> None:
    reader = _create_mock_reader([b"HTTP/1.1 200"])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_response_parser_wire_bytes_property() -> None:
    response_data = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    await parser.parse(reader)

    assert parser.wire_bytes == response_data


@pytest.mark.asyncio
async def test_async_response_parser_parse_with_read_exception() -> None:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(side_effect=OSError("Connection lost"))

    parser = AsyncResponseParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is None
    assert wire_bytes == b""


@pytest.mark.asyncio
async def test_async_response_parser_reset_during_eof_body_returns_none(
    response_with_partial_eof_body: bytes,
) -> None:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(
        side_effect=[
            response_with_partial_eof_body,
            ConnectionResetError("Connection reset"),
        ]
    )

    parser = AsyncResponseParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is None
    assert wire_bytes == response_with_partial_eof_body


@pytest.mark.asyncio
async def test_async_response_parser_head_completes_after_headers() -> None:
    response_data = b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n"
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader, request_method="HEAD")

    assert parsed is not None
    assert parsed.status_code == 200
    assert parsed.is_complete
    assert parsed.body == b""


@pytest.mark.asyncio
async def test_async_response_parser_parse_malformed_response() -> None:
    reader = _create_mock_reader([b"NOT A VALID HTTP RESPONSE\r\n"])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_response_parser_truncated_content_length_none() -> None:
    response_data = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/plain\r\n"
        b"Content-Length: 100\r\n"
        b"\r\n"
        b"Truncated!"
    )
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_response_parser_truncated_chunked_returns_none() -> None:
    response_data = (
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nHello\r\n"
    )
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_response_parser_lowercase_content_length_none() -> None:
    response_data = b"HTTP/1.1 200 OK\r\ncontent-length: 100\r\n\r\npartial"
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_response_parser_uppercase_transfer_encoding_none() -> (
    None
):
    response_data = (
        b"HTTP/1.1 200 OK\r\nTRANSFER-ENCODING: chunked\r\n\r\n5\r\nHello\r\n"
    )
    reader = _create_mock_reader([response_data])

    parser = AsyncResponseParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_multi_response_parser_buffers_large_body_parts() -> None:
    body = b"x" * (1024 * 1024)
    first_response = (
        b"HTTP/1.1 200 OK\r\n"
        + f"Content-Length: {len(body)}\r\n".encode()
        + b"\r\n"
        + body
    )
    second_response = b"HTTP/1.1 204 No Content\r\n\r\n"
    reader = _create_mock_reader([first_response + second_response])
    parser = AsyncMultiResponseParser(max_read=len(first_response) + 1024)

    first, first_wire = await parser.next_response(reader)
    second, second_wire = await parser.next_response(reader)

    assert first is not None
    assert first.body_parts == [body]
    assert first_wire == first_response
    assert second is not None
    assert second.status_code == 204
    assert second_wire == second_response


@pytest.mark.asyncio
async def test_multi_response_parser_parses_switching_protocols() -> None:
    response = (
        b"HTTP/1.1 101 Switching Protocols\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: websocket\r\n"
        b"\r\n"
    )
    reader = _create_mock_reader([response])
    parser = AsyncMultiResponseParser()

    parsed, wire = await parser.next_response(reader)

    assert parsed is not None
    assert parsed.status_code == 101
    assert parsed.is_complete
    assert wire == response


@pytest.mark.asyncio
async def test_multi_response_parser_streams_content_length_segments() -> None:
    headers = b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\n\r\n"
    reader = _create_mock_reader([headers + b"abc", b"def"])
    parser = AsyncMultiResponseParser()

    parsed, wire = await parser.next_response(reader)

    assert parsed is not None
    assert parsed.body_parts == [b"abc", b"def"]
    assert wire == headers + b"abcdef"


@pytest.mark.asyncio
async def test_multi_response_parser_preserves_chunked_trailers() -> None:
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Transfer-Encoding: gzip, chunked\r\n"
        b"\r\n"
        b"5;name=value\r\nHello\r\n"
        b"0\r\nX-Trailer: done\r\n\r\n"
    )
    reader = _create_mock_reader([response[:70], response[70:]])
    parser = AsyncMultiResponseParser()

    parsed, wire = await parser.next_response(reader)

    assert parsed is not None
    assert parsed.body == b"Hello"
    assert wire == response


@pytest.mark.asyncio
async def test_multi_response_parser_reads_many_chunked_segments() -> None:
    headers = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
    chunks = [b"1\r\nx\r\n"] * 1000
    terminator = b"0\r\n\r\n"
    reader = _create_mock_reader([headers, *chunks, terminator])
    parser = AsyncMultiResponseParser()

    parsed, wire = await parser.next_response(reader)

    assert parsed is not None
    assert parsed.body == b"x" * len(chunks)
    assert wire == headers + b"".join(chunks) + terminator


@pytest.mark.asyncio
async def test_multi_response_parser_reads_close_delimited_body() -> None:
    headers = b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n"
    reader = _create_mock_reader([headers + b"first", b"second"])
    parser = AsyncMultiResponseParser()

    parsed, wire = await parser.next_response(reader)

    assert parsed is not None
    assert parsed.body_parts == [b"first", b"second"]
    assert wire == headers + b"firstsecond"


@pytest.mark.asyncio
async def test_multi_response_parser_reset_during_eof_body_returns_none(
    response_with_partial_eof_body: bytes,
) -> None:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(
        side_effect=[
            response_with_partial_eof_body,
            ConnectionResetError("Connection reset"),
        ]
    )
    parser = AsyncMultiResponseParser()

    parsed, wire = await parser.next_response(reader)

    assert parsed is None
    assert wire == response_with_partial_eof_body


@pytest.mark.asyncio
async def test_multi_response_parser_head_retains_next_response() -> None:
    head_response = b"HTTP/1.1 200 OK\r\nContent-Length: 1000000\r\n\r\n"
    next_response = b"HTTP/1.1 204 No Content\r\n\r\n"
    reader = _create_mock_reader([head_response + next_response])
    parser = AsyncMultiResponseParser()

    head, head_wire = await parser.next_response(reader, request_method="head")
    following, following_wire = await parser.next_response(reader)

    assert head is not None
    assert head.body == b""
    assert head_wire == head_response
    assert following is not None
    assert following.status_code == 204
    assert following_wire == next_response


@pytest.mark.asyncio
async def test_multi_response_parser_truncated_body_returns_none() -> None:
    response = b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort"
    reader = _create_mock_reader([response])
    parser = AsyncMultiResponseParser()

    parsed, wire = await parser.next_response(reader)

    assert parsed is None
    assert wire == response


@pytest.mark.asyncio
async def test_multi_response_parser_invalid_chunk_size_returns_none() -> None:
    headers = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
    reader = _create_mock_reader([headers + b"Z"])
    parser = AsyncMultiResponseParser()

    parsed, wire = await parser.next_response(reader)

    assert parsed is None
    assert wire == headers + b"Z"


@pytest.fixture
def response_with_partial_eof_body() -> bytes:
    return b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\npartial"


def _create_mock_reader(data_chunks: list[bytes]) -> asyncio.StreamReader:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(side_effect=data_chunks + [b""])
    return reader


_BODYLESS_STATUS_CODES = (204, 304)
_STATUS_CODES = tuple(
    code for code in range(200, 600) if code not in _BODYLESS_STATUS_CODES
)
_RESPONSE_BODY_MODES = ("content-length", "chunked", "close-delimited")
_FRAMED_RESPONSE_BODY_MODES = ("content-length", "chunked")


@st.composite
def _response_wires(
    draw: st.DrawFn,
    modes: tuple[str, ...] = _RESPONSE_BODY_MODES,
) -> tuple[bytes, str]:
    status = draw(st.sampled_from(_STATUS_CODES))
    reason = draw(st.text(alphabet=HEADER_VALUE_ALPHABET, max_size=12))
    lines = [f"HTTP/1.1 {status} {reason}\r\n".encode("ascii")]
    lines.extend(
        f"{name}: {value}\r\n".encode("ascii")
        for name, value in draw(header_items())
    )
    mode = draw(st.sampled_from(modes))
    if mode == "content-length":
        body = draw(st.binary(max_size=64))
        lines.append(f"Content-Length: {len(body)}\r\n".encode("ascii"))
    elif mode == "chunked":
        lines.append(b"Transfer-Encoding: chunked\r\n")
        body = draw(chunked_bodies()).encoded
    else:
        body = draw(st.binary(max_size=64))
    lines.append(b"\r\n")
    return b"".join(lines) + body, mode


@st.composite
def _fragmented_response_cases(
    draw: st.DrawFn,
) -> tuple[bytes, str, list[bytes]]:
    encoded, mode = draw(_response_wires())
    return encoded, mode, draw(fragments_of(encoded))


@st.composite
def _pipelined_response_cases(
    draw: st.DrawFn,
) -> tuple[bytes, bytes, list[bytes]]:
    first, _ = draw(_response_wires(_FRAMED_RESPONSE_BODY_MODES))
    second, _ = draw(_response_wires(_FRAMED_RESPONSE_BODY_MODES))
    combined = first + second
    return first, second, draw(fragments_of(combined))


def _fragment_reader(fragments: list[bytes]) -> asyncio.StreamReader:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(
        side_effect=itertools.chain(fragments, itertools.repeat(b""))
    )
    return reader


async def _parse_with_single_parser(
    fragments: list[bytes],
) -> tuple[ParsedResponse | None, bytes]:
    parser = AsyncResponseParser()
    return await parser.parse(_fragment_reader(fragments))


async def _parse_with_multi_parser(
    fragments: list[bytes],
) -> tuple[ParsedResponse | None, bytes]:
    parser = AsyncMultiResponseParser()
    return await parser.next_response(_fragment_reader(fragments))


@pytest.mark.asyncio
@given(case=_fragmented_response_cases())
async def test_single_and_multi_response_parsers_agree_on_any_fragmentation(
    case: tuple[bytes, str, list[bytes]],
) -> None:
    encoded, mode, fragments = case
    single, single_wire = await _parse_with_single_parser(fragments)
    multi, multi_wire = await _parse_with_multi_parser(fragments)

    assert single is not None
    assert multi is not None
    assert single.is_complete
    assert multi.is_complete
    assert multi.status_code == single.status_code
    assert multi.status_text == single.status_text
    assert multi.http_version == single.http_version
    assert multi.headers == single.headers
    assert multi.body == single.body
    assert multi.is_eof_delimited == single.is_eof_delimited
    assert single_wire == encoded
    assert multi_wire == encoded
    assert single.is_eof_delimited == (mode == "close-delimited")


async def _check_pipelined_response_case(
    first_wire: bytes,
    second_wire: bytes,
    fragments: list[bytes],
) -> None:
    bytes_available = 0
    for fragment in fragments:
        bytes_available += len(fragment)
        if bytes_available >= len(first_wire):
            break
    first_read_crosses_boundary = bytes_available > len(first_wire)
    parser = AsyncMultiResponseParser()
    reader = _fragment_reader(fragments)

    first, actual_first_wire = await parser.next_response(reader)

    assert first is not None
    assert first.is_complete
    assert actual_first_wire == first_wire
    assert parser.has_buffered_data == first_read_crosses_boundary

    second, actual_second_wire = await parser.next_response(reader)

    assert second is not None
    assert second.is_complete
    assert actual_second_wire == second_wire
    assert not parser.has_buffered_data


@pytest.mark.asyncio
@given(case=_pipelined_response_cases())
async def test_multi_parser_preserves_pipelined_response_fragmentation(
    case: tuple[bytes, bytes, list[bytes]],
) -> None:
    await _check_pipelined_response_case(*case)


def test_multi_response_parser_has_no_buffered_data_initially() -> None:
    parser = AsyncMultiResponseParser()

    assert not parser.has_buffered_data


@pytest.mark.asyncio
async def test_multi_response_parser_no_buffered_data_after_exact_read() -> (
    None
):
    response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
    reader = _create_mock_reader([response])
    parser = AsyncMultiResponseParser()

    parsed, _ = await parser.next_response(reader)

    assert parsed is not None
    assert not parser.has_buffered_data


@pytest.mark.asyncio
async def test_multi_response_parser_has_buffered_data_after_extra_bytes() -> (
    None
):
    response = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nokEXTRA"
    reader = _create_mock_reader([response])
    parser = AsyncMultiResponseParser()

    parsed, _ = await parser.next_response(reader)

    assert parsed is not None
    assert parser.has_buffered_data
