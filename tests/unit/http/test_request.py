from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from localstub.http.headers import Headers
from localstub.http.request import (
    AsyncRequestParser,
    HTTPRequest,
    HTTPRequestHeaders,
    HTTPRequestReader,
    ParsedRequest,
    RecordedHTTPRequest,
    RequestProtocol,
)
from localstub.http.stream import take_unread_data


def test_parsed_request_default_values() -> None:
    request = ParsedRequest()

    assert request.method is None
    assert request.url is None
    assert request.http_version is None
    assert request.headers == []
    assert request.body_parts == []
    assert not request.is_complete
    assert not request.headers_complete


def test_parsed_request_body_property_empty() -> None:
    request = ParsedRequest()
    assert request.body == b""


def test_parsed_request_body_property_single_part() -> None:
    request = ParsedRequest(body_parts=[b"hello"])
    assert request.body == b"hello"


def test_parsed_request_body_property_multiple_parts() -> None:
    request = ParsedRequest(body_parts=[b"hello", b" ", b"world"])
    assert request.body == b"hello world"


def test_recorded_request_from_parsed_prefers_explicit_client() -> None:
    parsed = ParsedRequest(
        method="GET",
        url=b"/",
        http_version="1.1",
    )
    writer = Mock(spec=asyncio.StreamWriter)

    request = RecordedHTTPRequest.from_parsed(
        parsed,
        b"GET / HTTP/1.1\r\n\r\n",
        client=("127.0.0.1", 54321),
        writer=writer,
    )

    assert request.client == ("127.0.0.1", 54321)
    writer.get_extra_info.assert_not_called()


def test_request_protocol_on_message_begin_resets_result() -> None:
    protocol = RequestProtocol()
    protocol.result.method = "GET"
    protocol.result.url = b"/test"

    protocol.on_message_begin()

    assert protocol.result.method is None
    assert protocol.result.url is None


def test_request_protocol_on_url_sets_url() -> None:
    protocol = RequestProtocol()
    protocol.on_url(b"/api/users")

    assert protocol.result.url == b"/api/users"


def test_request_protocol_on_header_adds_headers() -> None:
    protocol = RequestProtocol()
    protocol.on_header(b"Content-Type", b"application/json")
    protocol.on_header(b"Host", b"localhost")

    assert len(protocol.result.headers) == 2
    assert protocol.result.headers[0] == (
        b"Content-Type",
        b"application/json",
    )
    assert protocol.result.headers[1] == (b"Host", b"localhost")


def test_request_protocol_on_body_adds_body_parts() -> None:
    protocol = RequestProtocol()
    protocol.on_body(b"chunk1")
    protocol.on_body(b"chunk2")

    assert protocol.result.body_parts == [b"chunk1", b"chunk2"]


def test_request_protocol_on_message_complete_sets_complete_flag() -> None:
    protocol = RequestProtocol()

    assert not protocol.result.is_complete

    protocol.on_message_complete()

    assert protocol.result.is_complete


def test_request_protocol_chunk_callbacks_are_noops() -> None:
    protocol = RequestProtocol()

    protocol.on_chunk_header()
    protocol.on_chunk_complete()


def test_request_protocol_preserves_completed_message() -> None:
    protocol = RequestProtocol()

    protocol.on_message_begin()
    protocol.on_url(b"/first")
    protocol.on_header(b"Host", b"first.com")
    protocol.on_body(b"first body")
    protocol.on_message_complete()

    assert protocol.result.is_complete
    assert protocol.result.url == b"/first"
    assert protocol.result.headers == [(b"Host", b"first.com")]
    assert protocol.result.body_parts == [b"first body"]

    protocol.on_message_begin()
    protocol.on_url(b"/second")
    protocol.on_header(b"Host", b"second.com")
    protocol.on_body(b"second body")
    protocol.on_message_complete()

    assert protocol.result.url == b"/first"
    assert protocol.result.headers == [(b"Host", b"first.com")]
    assert protocol.result.body_parts == [b"first body"]


def test_request_protocol_on_headers_complete_without_parser_is_noop() -> None:
    protocol = RequestProtocol()
    protocol.result.method = "GET"
    protocol.result.url = b"/test"

    protocol.on_headers_complete()

    assert protocol.result.method == "GET"
    assert protocol.result.url == b"/test"
    assert protocol.result.http_version is None


def test_request_protocol_on_url_accumulates_chunks() -> None:
    protocol = RequestProtocol()

    protocol.on_url(b"/api")
    protocol.on_url(b"/users")
    protocol.on_url(b"/123")

    assert protocol.result.url == b"/api/users/123"


@pytest.mark.asyncio
async def test_async_request_parser_parse_simple_get_request() -> None:
    request_data = b"GET /api/test HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = _create_mock_reader([request_data])

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.method == "GET"
    assert parsed.url == b"/api/test"
    assert parsed.http_version == "1.1"
    assert parsed.is_complete
    assert wire_bytes == request_data


@pytest.mark.asyncio
async def test_async_request_parser_parse_post_request_with_body() -> None:
    request_data = (
        b"POST /api/data HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 13\r\n"
        b"\r\n"
        b"Hello, World!"
    )
    reader = _create_mock_reader([request_data])

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.method == "POST"
    assert parsed.url == b"/api/data"
    assert parsed.body == b"Hello, World!"
    assert wire_bytes == request_data


@pytest.mark.asyncio
async def test_async_request_parser_parse_chunked_request() -> None:
    request_data = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nHello\r\n"
        b"6\r\n World\r\n"
        b"0\r\n\r\n"
    )
    reader = _create_mock_reader([request_data])

    parser = AsyncRequestParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is not None
    assert parsed.method == "POST"
    assert parsed.body == b"Hello World"
    assert parsed.is_complete


@pytest.mark.asyncio
async def test_async_request_parser_upgrade_content_length_body() -> None:
    request_data = (
        b"POST /chat HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: custom\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"hello"
    )
    reader = _create_mock_reader([request_data])

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.body == b"hello"
    assert parsed.is_complete
    assert wire_bytes == request_data


@pytest.mark.asyncio
async def test_async_request_parser_upgrade_chunked_body() -> None:
    request_data = (
        b"POST /chat HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: custom\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nHello\r\n"
        b"6\r\n World\r\n"
        b"0\r\n\r\n"
    )
    reader = _create_mock_reader([request_data])

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.body == b"Hello World"
    assert parsed.is_complete
    assert wire_bytes == request_data


@pytest.mark.asyncio
async def test_async_request_parser_upgrade_body_preserves_pipeline() -> None:
    smuggled = b"GET /admin HTTP/1.1\r\n\r\n"
    first_request = (
        b"POST /chat HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: custom\r\n"
        b"Content-Length: 23\r\n"
        b"\r\n" + smuggled
    )
    second_request = b"GET /next HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(first_request + second_request)
    reader.feed_eof()

    first, first_wire = await AsyncRequestParser().parse(reader)
    second, second_wire = await AsyncRequestParser().parse(reader)

    assert first is not None
    assert first.body == smuggled
    assert first_wire == first_request
    assert second is not None
    assert second.url == b"/next"
    assert second_wire == second_request


@pytest.mark.asyncio
async def test_async_request_parser_upgrade_without_body_completes() -> None:
    request_data = (
        b"GET /chat HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: websocket\r\n"
        b"\r\n"
    )
    protocol_bytes = b"upgraded-protocol-data"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data + protocol_bytes)
    reader.feed_eof()

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.is_complete
    assert not parsed.body_parts
    assert wire_bytes == request_data
    assert take_unread_data(reader) == protocol_bytes


@pytest.mark.asyncio
async def test_async_request_parser_upgrade_chunked_incremental() -> None:
    header_data = (
        b"POST /chat HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: custom\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
    )
    reader = _create_mock_reader([
        header_data,
        b"5\r\nHel",
        b"lo\r\n",
        b"0\r\n\r\n",
    ])

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.body == b"Hello"
    assert parsed.is_complete
    assert wire_bytes == header_data + b"5\r\nHello\r\n0\r\n\r\n"


@pytest.mark.asyncio
async def test_async_request_parser_upgrade_chunked_trailers() -> None:
    request_data = (
        b"POST /chat HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: custom\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nhello\r\n"
        b"0\r\n"
        b"X-Checksum: abc\r\n"
        b"\r\n"
    )
    reader = _create_mock_reader([request_data])

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.body == b"hello"
    assert parsed.is_complete
    assert wire_bytes == request_data


@pytest.mark.asyncio
async def test_async_request_parser_upgrade_zero_content_length() -> None:
    request_data = (
        b"POST /chat HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: custom\r\n"
        b"Content-Length: 0\r\n"
        b"\r\n"
    )
    reader = _create_mock_reader([request_data])

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.is_complete
    assert not parsed.body_parts
    assert wire_bytes == request_data


@pytest.mark.asyncio
async def test_async_request_parser_connect_ignores_content_length() -> None:
    request_data = (
        b"CONNECT example.com:443 HTTP/1.1\r\n"
        b"Host: example.com:443\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
    )
    tunnel_bytes = b"\x16\x03\x01ab"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data + tunnel_bytes)
    reader.feed_eof()

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is not None
    assert parsed.method == "CONNECT"
    assert not parsed.body_parts
    assert wire_bytes == request_data
    assert take_unread_data(reader) == tunnel_bytes


@pytest.mark.asyncio
async def test_async_request_parser_parse_with_connection_wire() -> None:
    request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = _create_mock_reader([request_data])
    connection_wire = bytearray()

    parser = AsyncRequestParser()
    await parser.parse(reader, connection_wire)

    assert bytes(connection_wire) == request_data


@pytest.mark.asyncio
async def test_async_request_parser_parse_with_connection_wire_pipeline() -> (
    None
):
    first_request = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"hello"
    )
    second_request = b"GET /next HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(first_request + second_request)
    connection_wire = bytearray()

    first_parser = AsyncRequestParser()
    first_parsed, first_wire = await first_parser.parse(
        reader,
        connection_wire,
    )

    second_parser = AsyncRequestParser()
    second_parsed, second_wire = await second_parser.parse(
        reader,
        connection_wire,
    )

    assert first_parsed is not None
    assert first_parsed.body == b"hello"
    assert first_wire == first_request

    assert second_parsed is not None
    assert second_parsed.url == b"/next"
    assert second_wire == second_request

    assert bytes(connection_wire) == first_request + second_request


@pytest.mark.asyncio
async def test_async_request_parser_parse_eof_before_complete() -> None:
    reader = _create_mock_reader([b"GET / HTTP/1.1\r\nHost:"])

    parser = AsyncRequestParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_request_parser_parse_malformed_request() -> None:
    reader = _create_mock_reader([b"NOT VALID HTTP AT ALL"])

    parser = AsyncRequestParser()
    parsed, _ = await parser.parse(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_request_parser_parse_malformed_single_byte_request() -> (
    None
):
    reader = _create_mock_reader([b"\x00"])

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is None
    assert wire_bytes == b"\x00"
    assert take_unread_data(reader) == b""


@pytest.mark.asyncio
async def test_async_request_parser_wire_bytes_property() -> None:
    request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = _create_mock_reader([request_data])

    parser = AsyncRequestParser()
    await parser.parse(reader)

    assert parser.wire_bytes == request_data


@pytest.mark.asyncio
async def test_async_request_parser_preserves_first_pipelined_request() -> (
    None
):
    first_request = b"GET /first HTTP/1.1\r\nHost: first.com\r\n\r\n"
    second_request = b"GET /second HTTP/1.1\r\nHost: second.com\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(first_request + second_request)

    parser1 = AsyncRequestParser()
    parsed1, wire_bytes1 = await parser1.parse(reader)

    assert parsed1 is not None
    assert parsed1.method == "GET"
    assert parsed1.url == b"/first"
    assert parsed1.is_complete
    host_value = next(h[1] for h in parsed1.headers if h[0] == b"Host")
    assert host_value == b"first.com"
    assert wire_bytes1 == first_request

    reader.feed_eof()

    parser2 = AsyncRequestParser()
    parsed2, wire_bytes2 = await parser2.parse(reader)

    assert parsed2 is not None
    assert parsed2.method == "GET"
    assert parsed2.url == b"/second"
    assert parsed2.is_complete
    host_value2 = next(h[1] for h in parsed2.headers if h[0] == b"Host")
    assert host_value2 == b"second.com"
    assert wire_bytes2 == second_request


@pytest.mark.asyncio
async def test_async_request_parser_preserves_pipeline_across_handoff() -> (
    None
):
    first_request = b"GET /first HTTP/1.1\r\nHost: first.com\r\n\r\n"
    second_request = b"GET /second HTTP/1.1\r\nHost: second.com\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(first_request + second_request)
    reader.feed_eof()

    first, _ = await AsyncRequestParser().parse(reader)
    second, _ = await AsyncRequestParser().parse(reader)

    assert first is not None
    assert first.url == b"/first"
    assert second is not None
    assert second.url == b"/second"


@pytest.mark.asyncio
async def test_async_request_parser_parse_with_read_exception() -> None:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(
        side_effect=ConnectionResetError("Connection reset")
    )

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is None
    assert wire_bytes == b""


@pytest.mark.asyncio
async def test_async_request_parser_preserves_remaining_buffer() -> None:
    reader = asyncio.StreamReader()
    malformed_with_extra = b"INVALID HTTP\x00\x01\x02extra data here"
    reader.feed_data(malformed_with_extra)

    parser = AsyncRequestParser()
    parsed, wire_bytes = await parser.parse(reader)

    assert parsed is None
    assert len(wire_bytes) > 0
    assert take_unread_data(reader)


@pytest.mark.asyncio
async def test_async_request_parser_parse_headers_stops_after_headers() -> (
    None
):
    request_data = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"hello"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    parser = AsyncRequestParser()
    parsed, header_wire, _ = await parser.parse_headers(reader)

    assert parsed is not None
    assert parsed.method == "POST"
    assert parsed.url == b"/upload"
    assert parsed.headers_complete
    assert parsed.body == b""
    assert not parsed.is_complete
    assert b"hello" not in header_wire
    assert b"\r\n\r\n" in header_wire


@pytest.mark.asyncio
async def test_async_request_parser_parse_headers_sets_complete_flag() -> None:
    request_data = b"GET /test HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    parser = AsyncRequestParser()
    parsed, _, _ = await parser.parse_headers(reader)

    assert parsed is not None
    assert parsed.headers_complete
    assert parsed.method == "GET"


@pytest.mark.asyncio
async def test_async_request_parser_headers_return_remaining_buffer() -> None:
    request_data = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"hello"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    parser = AsyncRequestParser()
    parsed, _, remaining = await parser.parse_headers(reader)

    assert parsed is not None
    assert len(remaining) > 0 or not reader.at_eof()


@pytest.mark.asyncio
async def test_async_request_parser_continue_body_completes_request() -> None:
    request_data = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"hello"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    parser = AsyncRequestParser()
    parsed, _, remaining = await parser.parse_headers(reader)

    assert parsed is not None
    assert not parsed.is_complete

    parsed, wire_bytes = await parser.continue_parse_body(reader, remaining)

    assert parsed is not None
    assert parsed.is_complete
    assert parsed.body == b"hello"
    assert wire_bytes == request_data


@pytest.mark.asyncio
async def test_async_request_parser_parse_headers_with_connection_wire() -> (
    None
):
    request_data = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"hello"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()
    connection_wire = bytearray()

    parser = AsyncRequestParser()
    parsed, _, _ = await parser.parse_headers(reader, connection_wire)

    assert parsed is not None
    assert len(connection_wire) > 0
    assert b"POST /upload" in bytes(connection_wire)


@pytest.mark.asyncio
async def test_async_request_parser_continue_body_with_connection_wire() -> (
    None
):
    request_data = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"hello"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()
    connection_wire = bytearray()

    parser = AsyncRequestParser()
    _, _, remaining = await parser.parse_headers(reader, connection_wire)

    parsed, _ = await parser.continue_parse_body(
        reader,
        remaining,
        connection_wire,
    )

    assert parsed is not None
    assert parsed.is_complete
    assert bytes(connection_wire) == request_data


@pytest.mark.asyncio
async def test_async_request_parser_parse_headers_eof_before_complete() -> (
    None
):
    reader = asyncio.StreamReader()
    reader.feed_data(b"GET / HTTP/1.1\r\nHost:")
    reader.feed_eof()

    parser = AsyncRequestParser()
    parsed, _, _ = await parser.parse_headers(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_request_parser_parse_headers_malformed_request() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"INVALID HTTP DATA")

    parser = AsyncRequestParser()
    parsed, _, _ = await parser.parse_headers(reader)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_request_parser_parse_headers_with_read_exception() -> (
    None
):
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(
        side_effect=ConnectionResetError("Connection reset")
    )

    parser = AsyncRequestParser()
    parsed, wire_bytes, remaining = await parser.parse_headers(reader)

    assert parsed is None
    assert wire_bytes == b""
    assert remaining == bytearray()


@pytest.mark.asyncio
async def test_async_request_parser_parse_headers_malformed_byte() -> None:
    reader = _create_mock_reader([b"\x00"])

    parser = AsyncRequestParser()
    parsed, wire_bytes, remaining = await parser.parse_headers(reader)

    assert parsed is None
    assert wire_bytes == b"\x00"
    assert remaining == bytearray()
    assert take_unread_data(reader) == b""


@pytest.mark.asyncio
async def test_async_request_parser_continue_body_eof_before_complete() -> (
    None
):
    request_data = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 100\r\n"
        b"\r\n"
        b"short"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    parser = AsyncRequestParser()
    parsed, _, remaining = await parser.parse_headers(reader)

    assert parsed is not None

    parsed, _ = await parser.continue_parse_body(reader, remaining)

    assert parsed is None


@pytest.mark.asyncio
async def test_async_request_parser_continue_body_preserves_pipeline() -> None:
    header_bytes = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
    )
    next_request = b"GET /next HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(
        side_effect=[header_bytes, b"hello" + next_request, b""]
    )

    parser = AsyncRequestParser()
    parsed, _, remaining = await parser.parse_headers(reader)

    assert parsed is not None
    assert remaining == bytearray()

    parsed, wire_bytes = await parser.continue_parse_body(reader, remaining)

    assert parsed is not None
    assert parsed.is_complete
    assert parsed.body == b"hello"
    assert wire_bytes == header_bytes + b"hello"
    assert take_unread_data(reader) == next_request


@pytest.mark.asyncio
async def test_async_request_parser_continue_body_with_read_exception() -> (
    None
):
    header_bytes = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
    )
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(
        side_effect=[header_bytes, ConnectionResetError("Connection reset")]
    )

    parser = AsyncRequestParser()
    parsed, _, remaining = await parser.parse_headers(reader)

    assert parsed is not None

    parsed, wire_bytes = await parser.continue_parse_body(reader, remaining)

    assert parsed is None
    assert wire_bytes == header_bytes


@pytest.mark.asyncio
async def test_async_request_parser_bad_chunk_preserves_remaining() -> None:
    header_bytes = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
    )
    next_request = b"GET /next HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(
        side_effect=[header_bytes, b"Z\r\nbroken\r\n" + next_request, b""]
    )

    parser = AsyncRequestParser()
    parsed, _, remaining = await parser.parse_headers(reader)

    assert parsed is not None

    parsed, wire_bytes = await parser.continue_parse_body(reader, remaining)

    assert parsed is None
    assert wire_bytes == header_bytes + b"Z"
    held = take_unread_data(reader)
    assert held.startswith(b"\r\nbroken\r\n")
    assert next_request in held


@pytest.mark.asyncio
async def test_async_request_parser_bad_chunk_without_remaining_bytes() -> (
    None
):
    header_bytes = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
    )
    reader = _create_mock_reader([header_bytes, b"Z"])

    parser = AsyncRequestParser()
    parsed, _, remaining = await parser.parse_headers(reader)

    assert parsed is not None

    parsed, wire_bytes = await parser.continue_parse_body(reader, remaining)

    assert parsed is None
    assert wire_bytes == header_bytes + b"Z"
    assert take_unread_data(reader) == b""


@pytest.mark.asyncio
async def test_http_request_reader_read_request_simple_get() -> None:
    request_data = b"GET /api/test HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader)

    assert request is not None
    assert request.method == "GET"
    assert request.target == "/api/test"
    assert request.http_version == "1.1"
    assert request.wire_raw_bytes == request_data


@pytest.mark.asyncio
async def test_http_request_reader_read_request_with_body() -> None:
    request_data = (
        b"POST /api/data HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: 13\r\n"
        b"\r\n"
        b"Hello, World!"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader)

    assert request is not None
    assert request.method == "POST"
    assert request.body == b"Hello, World!"
    assert request.text == "Hello, World!"


@pytest.mark.asyncio
async def test_http_request_reader_upgrade_request_records_body() -> None:
    request_data = (
        b"POST /chat HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Connection: Upgrade\r\n"
        b"Upgrade: custom\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"hello"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader)

    assert request is not None
    assert request.body == b"hello"
    assert request.wire_raw_bytes == request_data
    assert request.wire_body_bytes == b"hello"


@pytest.mark.asyncio
async def test_http_request_reader_preserves_pipeline_with_new_helpers() -> (
    None
):
    max_read = 8192
    second_request = b"GET /second HTTP/1.1\r\nHost: localhost\r\n\r\n"
    third_request = b"GET /third HTTP/1.1\r\nHost: localhost\r\n\r\n"
    first_template = (
        b"GET /first HTTP/1.1\r\nHost: localhost\r\nX-Padding: \r\n\r\n"
    )
    padding = b"x" * (max_read - len(first_template) - len(second_request))
    first_request = first_template.replace(
        b"X-Padding: ",
        b"X-Padding: " + padding,
    )
    reader = asyncio.StreamReader()
    reader.feed_data(first_request + second_request + third_request)
    reader.feed_eof()

    targets: list[str] = []
    for _ in range(3):
        request_reader = HTTPRequestReader(max_read=max_read)
        request = await request_reader.read_request(reader)
        assert request is not None
        targets.append(request.target)

    assert targets == ["/first", "/second", "/third"]


@pytest.mark.asyncio
async def test_http_request_reader_returns_none_on_parse_failure() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"NOT VALID HTTP")

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader)

    assert request is None


@pytest.mark.asyncio
async def test_http_request_reader_returns_none_on_eof() -> None:
    reader = asyncio.StreamReader()
    reader.feed_eof()

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader)

    assert request is None


@pytest.mark.asyncio
async def test_http_request_reader_with_connection_wire() -> None:
    request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()
    connection_wire = bytearray()

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(
        reader,
        connection_wire=connection_wire,
    )

    assert request is not None
    assert bytes(connection_wire) == request_data


@pytest.mark.asyncio
async def test_http_request_reader_extracts_client_info_from_writer() -> None:
    request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    writer = Mock(spec=asyncio.StreamWriter)
    writer.get_extra_info = Mock(return_value=("127.0.0.1", 54321))

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader, writer=writer)

    assert request is not None
    assert request.client == ("127.0.0.1", 54321)
    writer.get_extra_info.assert_called_once_with("peername")


@pytest.mark.asyncio
async def test_http_request_reader_invalid_peer_returns_none_client() -> None:
    request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    writer = Mock(spec=asyncio.StreamWriter)
    writer.get_extra_info = Mock(return_value=None)

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader, writer=writer)

    assert request is not None
    assert request.client is None


@pytest.mark.asyncio
async def test_http_request_reader_short_peer_tuple_returns_none_client() -> (
    None
):
    request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    writer = Mock(spec=asyncio.StreamWriter)
    writer.get_extra_info = Mock(return_value=("127.0.0.1",))

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader, writer=writer)

    assert request is not None
    assert request.client is None


@pytest.mark.asyncio
async def test_http_request_reader_without_writer_has_none_client() -> None:
    request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader, writer=None)

    assert request is not None
    assert request.client is None


@pytest.mark.asyncio
async def test_http_request_reader_with_custom_max_read() -> None:
    request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    http_reader = HTTPRequestReader(max_read=1024)
    request = await http_reader.read_request(reader)

    assert request is not None
    assert request.method == "GET"


def test_http_request_json_body_returns_none_for_missing_body() -> None:
    request = HTTPRequest(method="GET", target="/")
    assert request.json_body is None


def test_http_request_json_body_returns_none_for_empty_body() -> None:
    request = HTTPRequest(method="GET", target="/", body=b"")
    assert request.json_body is None


def test_http_request_json_body_parses_json() -> None:
    request = HTTPRequest(
        method="GET",
        target="/",
        body=b'{"key": "value"}',
    )
    assert request.json_body == {"key": "value"}


def test_http_request_equality_ignores_framing_headers() -> None:
    chunked = HTTPRequest(
        method="POST",
        target="/upload",
        headers=Headers.from_items([
            ("Transfer-Encoding", "chunked"),
            ("X-Api", "1"),
        ]),
        body=b"data",
    )
    sized = HTTPRequest(
        method="POST",
        target="/upload",
        headers=Headers.from_items([("Content-Length", "4"), ("X-Api", "1")]),
        body=b"data",
    )

    assert chunked == sized
    assert hash(chunked) == hash(sized)


def test_http_request_equality_compares_semantic_headers() -> None:
    first = HTTPRequest(
        method="POST",
        target="/upload",
        headers=Headers.from_items([("X-Api", "1")]),
    )
    second = HTTPRequest(
        method="POST",
        target="/upload",
        headers=Headers.from_items([("X-Api", "2")]),
    )

    assert first != second


def test_http_request_equality_distinguishes_absent_and_empty_body() -> None:
    absent = HTTPRequest(method="POST", target="/upload")
    empty = HTTPRequest(method="POST", target="/upload", body=b"")

    assert absent != empty


def test_http_request_equality_with_other_type_returns_not_equal() -> None:
    request = HTTPRequest(method="GET", target="/")

    assert request != "GET /"


def test_recorded_request_with_method_returns_updated_copy() -> None:
    request = HTTPRequest(method="GET", target="/")
    recorded = _make_recorded(request)

    updated = recorded.with_method("POST")

    assert updated.method == "POST"
    assert updated.target == "/"
    assert updated.as_received == request
    assert updated.wire_raw_bytes == recorded.wire_raw_bytes
    assert recorded.method == "GET"


def test_recorded_request_with_target_returns_updated_copy() -> None:
    request = HTTPRequest(method="GET", target="/")
    recorded = _make_recorded(request)

    updated = recorded.with_target("/updated")

    assert updated.target == "/updated"
    assert updated.method == "GET"
    assert updated.as_received == request
    assert updated.wire_raw_bytes == recorded.wire_raw_bytes
    assert recorded.target == "/"


def test_recorded_request_with_headers_returns_updated_copy() -> None:
    request = HTTPRequest(method="GET", target="/")
    recorded = _make_recorded(request)
    headers = Headers.from_items([("X-Test", "value")])

    updated = recorded.with_headers(headers)

    assert updated.headers == headers
    assert updated.as_received == request
    assert updated.wire_raw_bytes == recorded.wire_raw_bytes
    assert recorded.headers == Headers.empty()


def test_recorded_request_wire_body_bytes_returns_wire_body() -> None:
    wire_raw_bytes = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nhello\r\n"
        b"0\r\n\r\n"
    )
    request = HTTPRequest(
        method="POST",
        target="/upload",
        body=b"hello",
    )
    recorded = _make_recorded(request, wire_raw_bytes)

    assert recorded.wire_body_bytes == b"5\r\nhello\r\n0\r\n\r\n"


def test_recorded_request_wire_body_bytes_falls_back_to_body() -> None:
    request = HTTPRequest(
        method="POST",
        target="/upload",
        body=b"hello",
    )
    recorded = _make_recorded(request, b"not an http request")

    assert recorded.wire_body_bytes == b"hello"


def test_recorded_request_wire_body_bytes_empty_body_fallback() -> None:
    request = HTTPRequest(method="POST", target="/upload", body=b"")
    recorded = _make_recorded(request, b"not an http request")

    assert recorded.wire_body_bytes == b""


def test_recorded_request_wire_body_bytes_missing_body_fallback() -> None:
    request = HTTPRequest(method="POST", target="/upload")
    recorded = _make_recorded(request, b"not an http request")

    assert recorded.wire_body_bytes == b""


def test_http_request_exposes_immutable_headers() -> None:
    headers = Headers.from_items([("X-Test", "a")])
    request = HTTPRequest(
        method="GET",
        target="/",
        headers=headers,
    )

    assert request.headers["X-Test"] == "a"

    with pytest.raises(TypeError):
        request.headers["X-Test"] = "b"
    with pytest.raises(TypeError):
        del request.headers["X-Test"]


def test_http_request_headers_are_immutable() -> None:
    headers = Headers.from_items([("Host", "example.com")])

    partial = HTTPRequestHeaders(
        method="GET",
        path="/",
        http_version="1.1",
        headers=headers,
        wire_raw_bytes=b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n",
    )

    assert partial.headers["Host"] == "example.com"

    with pytest.raises(TypeError):
        partial.headers["Host"] = "other.example.com"


def test_http_request_is_proxy_request_true_for_http() -> None:
    request = HTTPRequest(
        method="GET",
        target="http://example.com/path",
    )
    assert request.is_proxy_request


def test_http_request_is_proxy_request_true_for_https() -> None:
    request = HTTPRequest(
        method="GET",
        target="https://example.com/path",
    )
    assert request.is_proxy_request


def test_http_request_is_proxy_request_false_for_origin_form() -> None:
    request = HTTPRequest(method="GET", target="/path")
    assert not request.is_proxy_request


def test_http_request_is_proxy_request_false_for_empty_target() -> None:
    request = HTTPRequest(method="GET", target="")
    assert not request.is_proxy_request


def test_http_request_target_uri_returns_parsed_for_absolute_uri() -> None:
    request = HTTPRequest(
        method="GET",
        target="http://example.com:8080/api?key=val",
    )
    uri = request.target_uri

    assert uri is not None
    assert uri.scheme == "http"
    assert uri.host == "example.com"
    assert uri.port == 8080
    assert uri.path == "/api?key=val"


def test_http_request_target_uri_returns_none_for_origin_form() -> None:
    request = HTTPRequest(method="GET", target="/path")
    assert request.target_uri is None


def test_http_request_target_uri_returns_none_for_empty_target() -> None:
    request = HTTPRequest(method="GET", target="")
    assert request.target_uri is None


def test_http_request_effective_path_extracts_absolute_path() -> None:
    request = HTTPRequest(
        method="GET",
        target="http://example.com/api/users?limit=10",
    )
    assert request.effective_path == "/api/users?limit=10"


def test_http_request_effective_path_returns_origin_form_path() -> None:
    request = HTTPRequest(
        method="GET",
        target="/api/users",
    )
    assert request.effective_path == "/api/users"


def test_http_request_effective_path_returns_slash_for_empty_target() -> None:
    request = HTTPRequest(method="GET", target="")
    assert request.effective_path == "/"


def test_wire_body_bytes_with_empty_wire_returns_empty() -> None:
    request = HTTPRequest(method="POST", target="/upload")
    recorded = _make_recorded(request, b"")

    assert recorded.wire_body_bytes == b""


def test_wire_body_bytes_with_bad_request_wire_returns_empty() -> None:
    request = HTTPRequest(method="POST", target="/upload")
    recorded = _make_recorded(request, b"\x00")

    assert recorded.wire_body_bytes == b""


def test_wire_body_bytes_with_incomplete_request_returns_empty() -> None:
    request = HTTPRequest(method="POST", target="/upload")
    recorded = _make_recorded(
        request,
        b"POST /upload HTTP/1.1\r\nHost: localhost\r\n",
    )

    assert recorded.wire_body_bytes == b""


@pytest.mark.asyncio
async def test_wire_body_bytes_preserves_chunked_wire_framing() -> None:
    request_data = (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nHello\r\n"
        b"6\r\n World\r\n"
        b"0\r\n\r\n"
    )
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    http_reader = HTTPRequestReader()
    request = await http_reader.read_request(reader)

    assert request is not None
    assert request.body == b"Hello World"
    assert request.wire_body_bytes == b"5\r\nHello\r\n6\r\n World\r\n0\r\n\r\n"


def _create_mock_reader(data_chunks: list[bytes]) -> asyncio.StreamReader:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(side_effect=data_chunks + [b""])
    return reader


def _make_recorded(
    request: HTTPRequest,
    wire_raw_bytes: bytes = b"GET / HTTP/1.1\r\n\r\n",
) -> RecordedHTTPRequest:
    return RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=wire_raw_bytes,
        http_version="1.1",
    )
