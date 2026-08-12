from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from localstub.http.headers import Headers
from localstub.http.response import (
    AsyncMultiResponseParser,
    AsyncResponseParser,
    ParsedResponse,
    RecordedHTTPResponse,
    ResponseProtocol,
)
from localstub.http.responsespec import HTTPResponse


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
