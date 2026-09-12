from __future__ import annotations

import asyncio
import itertools
import sys
from typing import Any
from unittest.mock import AsyncMock, Mock

import httptools
import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.http.framing import ChunkScanError
from localstub.http.headers import Headers
from localstub.http.request import (
    AsyncRequestParser,
    HTTPRequest,
    HTTPRequestHeaders,
    HTTPRequestReader,
    ParsedRequest,
    ParseOutcome,
    ParseStop,
    RecordedHTTPRequest,
    RequestProtocol,
)
from localstub.http.stream import take_unread_data

from .strategies import (
    chunked_bodies,
    fragments_of,
    header_items,
)

GZIP_HEAD = (
    b"POST / HTTP/1.1\r\nHost: localhost\r\nTransfer-Encoding: gzip\r\n\r\n"
)
GZIP_REQUEST = GZIP_HEAD + b"body"


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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    parsed, _ = _parsed(await parser.parse(reader))

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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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

    first, first_wire = _parsed(await AsyncRequestParser().parse(reader))
    second, second_wire = _parsed(await AsyncRequestParser().parse(reader))

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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    first_parsed, first_wire = _parsed(
        await first_parser.parse(
            reader,
            connection_wire,
        )
    )

    second_parser = AsyncRequestParser()
    second_parsed, second_wire = _parsed(
        await second_parser.parse(
            reader,
            connection_wire,
        )
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
    parsed, _ = _parsed(await parser.parse(reader))

    assert parsed is None


@pytest.mark.asyncio
async def test_async_request_parser_parse_malformed_request() -> None:
    reader = _create_mock_reader([b"NOT VALID HTTP AT ALL"])

    parser = AsyncRequestParser()
    parsed, _ = _parsed(await parser.parse(reader))

    assert parsed is None


@pytest.mark.asyncio
async def test_async_request_parser_parse_malformed_single_byte_request() -> (
    None
):
    reader = _create_mock_reader([b"\x00"])

    parser = AsyncRequestParser()
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    parsed1, wire_bytes1 = _parsed(await parser1.parse(reader))

    assert parsed1 is not None
    assert parsed1.method == "GET"
    assert parsed1.url == b"/first"
    assert parsed1.is_complete
    host_value = next(h[1] for h in parsed1.headers if h[0] == b"Host")
    assert host_value == b"first.com"
    assert wire_bytes1 == first_request

    reader.feed_eof()

    parser2 = AsyncRequestParser()
    parsed2, wire_bytes2 = _parsed(await parser2.parse(reader))

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

    first, _ = _parsed(await AsyncRequestParser().parse(reader))
    second, _ = _parsed(await AsyncRequestParser().parse(reader))

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
    parsed, wire_bytes = _parsed(await parser.parse(reader))

    assert parsed is None
    assert wire_bytes == b""


@pytest.mark.asyncio
async def test_async_request_parser_preserves_remaining_buffer() -> None:
    reader = asyncio.StreamReader()
    malformed_with_extra = b"INVALID HTTP\x00\x01\x02extra data here"
    reader.feed_data(malformed_with_extra)

    parser = AsyncRequestParser()
    parsed, wire_bytes = _parsed(await parser.parse(reader))

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
    parsed, header_wire, _ = _headers(await parser.parse_headers(reader))

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
    parsed, _, _ = _headers(await parser.parse_headers(reader))

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
    parsed, _, remaining = _headers(await parser.parse_headers(reader))

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
    parsed, _, remaining = _headers(await parser.parse_headers(reader))

    assert parsed is not None
    assert not parsed.is_complete

    parsed, wire_bytes = _parsed(
        await parser.continue_parse_body(reader, remaining)
    )

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
    parsed, _, _ = _headers(
        await parser.parse_headers(reader, connection_wire)
    )

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
    _, _, remaining = _headers(
        await parser.parse_headers(reader, connection_wire)
    )

    parsed, _ = _parsed(
        await parser.continue_parse_body(
            reader,
            remaining,
            connection_wire,
        )
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
    parsed, _, _ = _headers(await parser.parse_headers(reader))

    assert parsed is None


@pytest.mark.asyncio
async def test_async_request_parser_parse_headers_malformed_request() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"INVALID HTTP DATA")

    parser = AsyncRequestParser()
    parsed, _, _ = _headers(await parser.parse_headers(reader))

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
    parsed, wire_bytes, remaining = _headers(
        await parser.parse_headers(reader)
    )

    assert parsed is None
    assert wire_bytes == b""
    assert remaining == bytearray()


@pytest.mark.asyncio
async def test_async_request_parser_parse_headers_malformed_byte() -> None:
    reader = _create_mock_reader([b"\x00"])

    parser = AsyncRequestParser()
    parsed, wire_bytes, remaining = _headers(
        await parser.parse_headers(reader)
    )

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
    parsed, _, remaining = _headers(await parser.parse_headers(reader))

    assert parsed is not None

    parsed, _ = _parsed(await parser.continue_parse_body(reader, remaining))

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
    parsed, _, remaining = _headers(await parser.parse_headers(reader))

    assert parsed is not None
    assert remaining == bytearray()

    parsed, wire_bytes = _parsed(
        await parser.continue_parse_body(reader, remaining)
    )

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
    parsed, _, remaining = _headers(await parser.parse_headers(reader))

    assert parsed is not None

    parsed, wire_bytes = _parsed(
        await parser.continue_parse_body(reader, remaining)
    )

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
    parsed, _, remaining = _headers(await parser.parse_headers(reader))

    assert parsed is not None

    parsed, wire_bytes = _parsed(
        await parser.continue_parse_body(reader, remaining)
    )

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
    parsed, _, remaining = _headers(await parser.parse_headers(reader))

    assert parsed is not None

    parsed, wire_bytes = _parsed(
        await parser.continue_parse_body(reader, remaining)
    )

    assert parsed is None
    assert wire_bytes == header_bytes + b"Z"
    assert take_unread_data(reader) == b""


@pytest.mark.asyncio
async def test_async_request_parser_parse_rejected_transfer_encoding() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(GZIP_REQUEST)

    parser = AsyncRequestParser()
    outcome = await parser.parse(reader)

    assert outcome.stop is ParseStop.PARSE_ERROR
    assert isinstance(outcome.error, httptools.HttpParserError)
    assert outcome.parsed is not None
    assert outcome.parsed.headers_complete
    assert not outcome.parsed.is_complete
    assert outcome.complete_request is None
    assert outcome.wire_bytes == GZIP_HEAD
    assert take_unread_data(reader) == b"body"


@pytest.mark.asyncio
async def test_async_request_parser_continue_body_after_rejected_headers() -> (
    None
):
    reader = asyncio.StreamReader()
    reader.feed_data(GZIP_REQUEST)
    connection_wire = bytearray()

    parser = AsyncRequestParser()
    headers, remaining = await parser.parse_headers(reader, connection_wire)
    outcome = await parser.continue_parse_body(
        reader, remaining, connection_wire
    )

    assert headers.stop is ParseStop.PARSE_ERROR
    assert outcome.stop is ParseStop.PARSE_ERROR
    assert outcome.error is headers.error
    assert outcome.complete_request is None
    assert outcome.wire_bytes == GZIP_HEAD
    assert bytes(connection_wire) == GZIP_HEAD
    assert take_unread_data(reader) == b"body"


@pytest.mark.asyncio
async def test_async_request_parser_content_length_with_leading_zeros() -> (
    None
):
    zeros = b"0" * (sys.get_int_max_str_digits() + 1)
    reader = asyncio.StreamReader()
    reader.feed_data(
        b"POST / HTTP/1.1\r\nHost: localhost\r\n"
        b"Content-Length: " + zeros + b"5\r\n\r\nhello"
    )

    parser = AsyncRequestParser()
    parsed, _ = _parsed(await parser.parse(reader))

    assert parsed is not None
    assert parsed.body == b"hello"


@pytest.mark.asyncio
async def test_http_request_reader_bad_transfer_encoding_returns_none() -> (
    None
):
    reader = asyncio.StreamReader()
    reader.feed_data(GZIP_REQUEST)

    request = await HTTPRequestReader().read_request(reader)

    assert request is None


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
        _untyped(request.headers)["X-Test"] = "b"
    with pytest.raises(TypeError):
        del _untyped(request.headers)["X-Test"]


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
        _untyped(partial.headers)["Host"] = "other.example.com"


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


def _parsed(outcome: ParseOutcome) -> tuple[ParsedRequest | None, bytes]:
    """The complete request and wire bytes, the way callers consume them."""
    return outcome.complete_request, outcome.wire_bytes


def _headers(
    staged: tuple[ParseOutcome, bytearray],
) -> tuple[ParsedRequest | None, bytes, bytearray]:
    """The parsed headers, header wire bytes, and remaining buffer."""
    outcome, remaining = staged
    return outcome.parsed, outcome.wire_bytes, remaining


def _create_mock_reader(data_chunks: list[bytes]) -> asyncio.StreamReader:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(side_effect=data_chunks + [b""])
    return reader


def _untyped(value: object) -> Any:
    """Drop static typing to attempt an operation the types forbid."""
    return value


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


_METHODS = ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS")
_TARGET_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789/-_.?=&"
_BODY_MODES = ("none", "content-length", "chunked")


@st.composite
def _request_wires(draw: st.DrawFn) -> bytes:
    method = draw(st.sampled_from(_METHODS))
    target = "/" + draw(st.text(alphabet=_TARGET_ALPHABET, max_size=16))
    lines = [f"{method} {target} HTTP/1.1\r\n".encode("ascii")]
    lines.extend(
        f"{name}: {value}\r\n".encode("ascii")
        for name, value in draw(header_items())
    )
    mode = draw(st.sampled_from(_BODY_MODES))
    body = b""
    if mode == "content-length":
        body = draw(st.binary(max_size=64))
        lines.append(f"Content-Length: {len(body)}\r\n".encode("ascii"))
    elif mode == "chunked":
        lines.append(b"Transfer-Encoding: chunked\r\n")
        body = draw(chunked_bodies()).encoded
    lines.append(b"\r\n")
    return b"".join(lines) + body


@st.composite
def _fragmented_request_cases(draw: st.DrawFn) -> tuple[bytes, list[bytes]]:
    encoded = draw(_request_wires())
    return encoded, draw(fragments_of(encoded))


@st.composite
def _pipelined_request_cases(
    draw: st.DrawFn,
) -> tuple[list[bytes], list[bytes]]:
    requests = draw(st.lists(_request_wires(), min_size=2, max_size=3))
    return requests, draw(fragments_of(b"".join(requests)))


def _fragment_reader(fragments: list[bytes]) -> asyncio.StreamReader:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(
        side_effect=itertools.chain(fragments, itertools.repeat(b""))
    )
    return reader


async def _parse_one_request(
    fragments: list[bytes],
) -> tuple[ParsedRequest | None, bytes, bytes]:
    reader = _fragment_reader(fragments)
    parsed, wire = _parsed(await AsyncRequestParser().parse(reader))
    return parsed, wire, take_unread_data(reader)


async def _parse_request_sequence(
    count: int,
    fragments: list[bytes],
) -> tuple[list[tuple[ParsedRequest | None, bytes]], bytes]:
    reader = _fragment_reader(fragments)
    outcomes = [
        _parsed(await AsyncRequestParser().parse(reader)) for _ in range(count)
    ]
    return outcomes, take_unread_data(reader)


@pytest.mark.asyncio
@given(case=_fragmented_request_cases())
async def test_parser_with_any_fragmentation_matches_one_shot_parse(
    case: tuple[bytes, list[bytes]],
) -> None:
    encoded, fragments = case
    one_shot, one_shot_wire, one_shot_leftover = await _parse_one_request([
        encoded
    ])
    (
        fragmented,
        fragmented_wire,
        fragmented_leftover,
    ) = await _parse_one_request(fragments)

    assert one_shot is not None
    assert fragmented is not None
    assert one_shot.is_complete
    assert fragmented.is_complete
    assert fragmented.method == one_shot.method
    assert fragmented.url == one_shot.url
    assert fragmented.http_version == one_shot.http_version
    assert fragmented.headers == one_shot.headers
    assert fragmented.body == one_shot.body
    assert one_shot_wire == encoded
    assert fragmented_wire == encoded
    assert one_shot_leftover == b""
    assert fragmented_leftover == b""


@pytest.mark.asyncio
@given(case=_pipelined_request_cases())
async def test_parser_pipelined_stream_preserves_per_request_wire_bytes(
    case: tuple[list[bytes], list[bytes]],
) -> None:
    requests, fragments = case
    outcomes, leftover = await _parse_request_sequence(
        len(requests), fragments
    )

    for (parsed, wire), encoded in zip(outcomes, requests, strict=True):
        assert parsed is not None
        assert parsed.is_complete
        assert wire == encoded
    assert leftover == b""


_NEXT_REQUEST = b"GET /next HTTP/1.1\r\nHost: localhost\r\n\r\n"
_CHUNKED_HEADERS = (
    b"POST /upload HTTP/1.1\r\n"
    b"Host: localhost\r\n"
    b"Transfer-Encoding: chunked\r\n"
    b"\r\n"
)
_UPGRADE_CHUNKED_HEADERS = (
    b"POST /chat HTTP/1.1\r\n"
    b"Host: localhost\r\n"
    b"Connection: Upgrade\r\n"
    b"Upgrade: custom\r\n"
    b"Transfer-Encoding: chunked\r\n"
    b"\r\n"
)
_TWO_CHUNKS = b"3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n"
_BAD_EXTENSION_CHUNK = b"5;a\x01\r\nhello\r\n"


def _content_length_headers(length: int, *, upgrade: bool = False) -> bytes:
    upgrade_lines = b""
    if upgrade:
        upgrade_lines = b"Connection: Upgrade\r\nUpgrade: custom\r\n"
    return (
        b"POST /upload HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        + upgrade_lines
        + f"Content-Length: {length}\r\n\r\n".encode("ascii")
    )


class _ScriptedReader:
    """A byte stream whose reads return scripted chunks or raise."""

    def __init__(self, script: list[bytes | Exception]) -> None:
        self._script = script

    async def read(self, n: int = -1) -> bytes:
        if not self._script:
            return b""
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        if 0 <= n < len(item):
            self._script.insert(0, item[n:])
            return item[:n]
        return item


def _partial(outcome: ParseOutcome) -> ParsedRequest:
    """The parsed request of an outcome whose headers completed."""
    assert outcome.parsed is not None
    return outcome.parsed


async def _parse_body(
    script: list[bytes | Exception],
    *,
    max_body_bytes: int | None = None,
) -> tuple[ParseOutcome, bytes]:
    """Parse headers then the body; return the outcome and unread bytes."""
    reader = _ScriptedReader(script)
    parser = AsyncRequestParser()
    staged, remaining = await parser.parse_headers(reader)
    assert staged.parsed is not None
    outcome = await parser.continue_parse_body(
        reader,
        remaining,
        max_body_bytes=max_body_bytes,
    )
    return outcome, take_unread_data(reader)


def test_http_request_text_without_body_returns_empty_string() -> None:
    request = HTTPRequest(method="GET", target="/")

    assert request.text == ""


def test_recorded_request_json_body_delegates_to_request() -> None:
    request = HTTPRequest(method="POST", target="/", body=b'{"key": 1}')

    assert _make_recorded(request).json_body == {"key": 1}


def test_recorded_request_is_proxy_request_delegates_to_request() -> None:
    request = HTTPRequest(method="GET", target="http://example.com/path")

    assert _make_recorded(request).is_proxy_request


def test_recorded_request_body_complete_defaults_to_true() -> None:
    recorded = _make_recorded(HTTPRequest(method="GET", target="/"))

    assert recorded.body_complete


def test_from_parsed_complete_message_marks_body_complete() -> None:
    parsed = ParsedRequest(
        method="POST",
        url=b"/upload",
        http_version="1.1",
        headers=[(b"Content-Length", b"5")],
        body_parts=[b"hello"],
        is_complete=True,
        headers_complete=True,
    )

    recorded = RecordedHTTPRequest.from_parsed(
        parsed, _content_length_headers(5) + b"hello"
    )

    assert recorded.body_complete
    assert recorded.body == b"hello"


def test_from_parsed_incomplete_message_marks_body_incomplete() -> None:
    parsed = ParsedRequest(
        method="POST",
        url=b"/upload",
        http_version="1.1",
        headers=[(b"Content-Length", b"10")],
        body_parts=[b"short"],
        headers_complete=True,
    )

    recorded = RecordedHTTPRequest.from_parsed(
        parsed, _content_length_headers(10) + b"short"
    )

    assert not recorded.body_complete
    assert recorded.body == b"short"
    assert recorded.wire_body_bytes == b"short"


def test_from_parsed_partial_without_payload_has_empty_body() -> None:
    parsed = ParsedRequest(
        method="POST",
        url=b"/upload",
        http_version="1.1",
        headers=[(b"Content-Length", b"10")],
        headers_complete=True,
    )

    recorded = RecordedHTTPRequest.from_parsed(
        parsed, _content_length_headers(10)
    )

    assert not recorded.body_complete
    assert recorded.body == b""


def test_from_parsed_chunked_without_payload_has_empty_body() -> None:
    parsed = ParsedRequest(
        method="POST",
        url=b"/upload",
        http_version="1.1",
        headers=[(b"Transfer-Encoding", b"chunked")],
        is_complete=True,
        headers_complete=True,
    )

    recorded = RecordedHTTPRequest.from_parsed(
        parsed, _CHUNKED_HEADERS + b"0\r\n\r\n"
    )

    assert recorded.body == b""


def test_parse_outcome_complete_request_is_none_before_headers() -> None:
    outcome = ParseOutcome(
        parsed=None, wire_bytes=b"", stop=ParseStop.COMPLETE
    )

    assert outcome.complete_request is None
    assert outcome.error is None


def test_parse_outcome_complete_request_is_none_for_non_complete_stop() -> (
    None
):
    parsed = ParsedRequest(headers_complete=True, is_complete=True)
    outcome = ParseOutcome(parsed=parsed, wire_bytes=b"", stop=ParseStop.EOF)

    assert outcome.complete_request is None


def test_parse_outcome_complete_request_is_none_for_incomplete_message() -> (
    None
):
    parsed = ParsedRequest(headers_complete=True)
    outcome = ParseOutcome(
        parsed=parsed, wire_bytes=b"", stop=ParseStop.COMPLETE
    )

    assert outcome.complete_request is None


def test_parse_outcome_complete_request_returns_completed_message() -> None:
    parsed = ParsedRequest(headers_complete=True, is_complete=True)
    outcome = ParseOutcome(
        parsed=parsed, wire_bytes=b"", stop=ParseStop.COMPLETE
    )

    assert outcome.complete_request is parsed


@pytest.mark.asyncio
async def test_parse_idle_eof_stops_with_eof_and_nothing_read() -> None:
    reader = asyncio.StreamReader()
    reader.feed_eof()

    outcome = await AsyncRequestParser().parse(reader)

    assert outcome.stop is ParseStop.EOF
    assert outcome.parsed is None
    assert outcome.wire_bytes == b""
    assert outcome.error is None


@pytest.mark.asyncio
async def test_parse_eof_mid_headers_stops_with_eof_and_header_prefix() -> (
    None
):
    partial_headers = b"GET / HTTP/1.1\r\nHost:"
    reader = asyncio.StreamReader()
    reader.feed_data(partial_headers)
    reader.feed_eof()

    outcome = await AsyncRequestParser().parse(reader)

    assert outcome.stop is ParseStop.EOF
    assert outcome.parsed is None
    assert outcome.wire_bytes == partial_headers


@pytest.mark.asyncio
async def test_parse_eof_mid_body_stops_with_eof_and_partial_request() -> None:
    headers = _content_length_headers(10)
    reader = asyncio.StreamReader()
    reader.feed_data(headers + b"short")
    reader.feed_eof()

    outcome = await AsyncRequestParser().parse(reader)

    assert outcome.stop is ParseStop.EOF
    assert _partial(outcome).body == b"short"
    assert not _partial(outcome).is_complete
    assert outcome.wire_bytes == headers + b"short"
    assert outcome.complete_request is None


@pytest.mark.asyncio
async def test_parse_header_parse_error_stops_with_parser_exception() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"NOT VALID HTTP\r\n\r\n")
    reader.feed_eof()

    outcome = await AsyncRequestParser().parse(reader)

    assert outcome.stop is ParseStop.PARSE_ERROR
    assert outcome.parsed is None
    assert isinstance(outcome.error, httptools.HttpParserError)


@pytest.mark.asyncio
async def test_parse_chunk_framing_error_stops_with_partial_request() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(_CHUNKED_HEADERS + b"5\r\nhello\r\nZ\r\n")
    reader.feed_eof()

    outcome = await AsyncRequestParser().parse(reader)

    assert outcome.stop is ParseStop.PARSE_ERROR
    assert _partial(outcome).body == b"hello"
    assert not _partial(outcome).is_complete
    assert isinstance(outcome.error, ChunkScanError)
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"5\r\nhello\r\nZ"
    assert take_unread_data(reader) == b"\r\n"


@pytest.mark.asyncio
async def test_parse_read_error_stops_with_read_error_and_exception() -> None:
    error = ConnectionResetError("Connection reset")
    reader = _ScriptedReader([error])

    outcome = await AsyncRequestParser().parse(reader)

    assert outcome.stop is ParseStop.READ_ERROR
    assert outcome.parsed is None
    assert outcome.error is error
    assert outcome.wire_bytes == b""


@pytest.mark.asyncio
async def test_parse_read_error_mid_body_keeps_partial_request() -> None:
    headers = _content_length_headers(10)
    error = ConnectionResetError("Connection reset")
    reader = _ScriptedReader([headers + b"short", error])

    outcome = await AsyncRequestParser().parse(reader)

    assert outcome.stop is ParseStop.READ_ERROR
    assert _partial(outcome).body == b"short"
    assert outcome.error is error
    assert outcome.wire_bytes == headers + b"short"


@pytest.mark.asyncio
async def test_parse_headers_repeated_leading_crlf_is_parse_error() -> None:
    request_data = b"\r\n\r\nGET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
    reader = asyncio.StreamReader()
    reader.feed_data(request_data)
    reader.feed_eof()

    outcome, remaining = await AsyncRequestParser().parse_headers(reader)

    assert outcome.stop is ParseStop.PARSE_ERROR
    assert outcome.parsed is None
    assert outcome.wire_bytes == request_data
    assert remaining == bytearray()


@pytest.mark.asyncio
async def test_continue_parse_body_zero_threshold_consumes_no_body() -> None:
    headers = _content_length_headers(5)

    outcome, leftover = await _parse_body(
        [headers + b"hello"], max_body_bytes=0
    )

    assert outcome.stop is ParseStop.COMPLETE
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b""
    assert outcome.complete_request is None
    assert outcome.wire_bytes == headers
    assert leftover == b"hello"


@pytest.mark.asyncio
async def test_continue_parse_body_threshold_below_length_stops_in_body() -> (
    None
):
    headers = _content_length_headers(5)

    outcome, leftover = await _parse_body(
        [headers + b"hello"], max_body_bytes=2
    )

    assert outcome.stop is ParseStop.COMPLETE
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"he"
    assert outcome.complete_request is None
    assert outcome.wire_bytes == headers + b"he"
    assert leftover == b"llo"


@pytest.mark.asyncio
async def test_continue_parse_body_threshold_equal_to_length_completes() -> (
    None
):
    headers = _content_length_headers(5)

    outcome, leftover = await _parse_body(
        [headers + b"hello"], max_body_bytes=5
    )

    assert outcome.complete_request is not None
    assert outcome.complete_request.body == b"hello"
    assert outcome.wire_bytes == headers + b"hello"
    assert leftover == b""


@pytest.mark.asyncio
async def test_continue_parse_body_threshold_above_length_completes() -> None:
    headers = _content_length_headers(5)

    outcome, leftover = await _parse_body(
        [headers + b"hello"], max_body_bytes=10
    )

    assert outcome.complete_request is not None
    assert outcome.complete_request.body == b"hello"
    assert leftover == b""


@pytest.mark.asyncio
async def test_continue_parse_body_bodyless_request_ignores_threshold() -> (
    None
):
    request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"

    outcome, leftover = await _parse_body([request_data], max_body_bytes=0)

    assert outcome.stop is ParseStop.COMPLETE
    assert outcome.complete_request is not None
    assert outcome.complete_request.body_parts == []
    assert outcome.wire_bytes == request_data
    assert leftover == b""


@pytest.mark.asyncio
async def test_continue_parse_body_eof_under_threshold_stops_with_eof() -> (
    None
):
    headers = _content_length_headers(10)

    outcome, leftover = await _parse_body(
        [headers + b"short"], max_body_bytes=8
    )

    assert outcome.stop is ParseStop.EOF
    assert _partial(outcome).body == b"short"
    assert not _partial(outcome).is_complete
    assert outcome.wire_bytes == headers + b"short"
    assert leftover == b""


@pytest.mark.asyncio
async def test_continue_parse_body_threshold_mid_read_keeps_surplus() -> None:
    headers = _content_length_headers(8)

    outcome, leftover = await _parse_body(
        [headers, b"hel", b"lo w", b"or"], max_body_bytes=5
    )

    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"hello"
    assert outcome.wire_bytes == headers + b"hello"
    assert leftover == b" w"


@pytest.mark.asyncio
async def test_continue_parse_body_pipelined_request_stops_at_boundary() -> (
    None
):
    headers = _content_length_headers(5)

    outcome, leftover = await _parse_body(
        [headers + b"hello" + _NEXT_REQUEST], max_body_bytes=100
    )

    assert outcome.complete_request is not None
    assert outcome.complete_request.body == b"hello"
    assert outcome.wire_bytes == headers + b"hello"
    assert leftover == _NEXT_REQUEST


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "prefix"),
    [
        (b"8\r\nabcdefghX\n", b"8\r\nab"),
        (b"8\r\nabcdefgh\rX", b"8\r\nab"),
        (b"8\r\nabcdefgh\r\nZ\r\n", b"8\r\nab"),
        (b"1\r\na\r\n7\r\nbcdefghX\n", b"1\r\na\r\n7\r\nb"),
    ],
)
@pytest.mark.parametrize(
    "headers", [_CHUNKED_HEADERS, _UPGRADE_CHUNKED_HEADERS]
)
async def test_body_budget_precedes_later_chunk_errors_across_fragments(
    body: bytes,
    prefix: bytes,
    headers: bytes,
) -> None:
    for split in range(len(body)):
        script = [headers + body[:split], body[split:]]
        outcome, leftover = await _parse_body(script, max_body_bytes=2)

        assert outcome.stop is ParseStop.COMPLETE
        assert outcome.error is None
        assert not _partial(outcome).is_complete
        assert _partial(outcome).body == b"ab"
        assert outcome.wire_bytes == headers + prefix
        if split == 0:
            assert leftover == body[len(prefix) :]


@pytest.mark.parametrize("budget", [2, 3])
@pytest.mark.asyncio
async def test_chunk_error_after_complete_chunk_respects_budget(
    budget: int,
) -> None:
    outcome, leftover = await _parse_body(
        [_CHUNKED_HEADERS + b"2\r\nab\r\nZ\r\n"], max_body_bytes=budget
    )

    assert _partial(outcome).body == b"ab"
    if budget == 2:
        assert outcome.stop is ParseStop.COMPLETE
        assert outcome.error is None
        assert outcome.wire_bytes == _CHUNKED_HEADERS + b"2\r\nab\r\n"
        assert leftover == b"Z\r\n"
    else:
        assert outcome.stop is ParseStop.PARSE_ERROR
        assert isinstance(outcome.error, ChunkScanError)
        assert outcome.wire_bytes == _CHUNKED_HEADERS + b"2\r\nab\r\nZ"
        assert leftover == b"\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_stop_within_chunk() -> None:
    outcome, leftover = await _parse_body(
        [_CHUNKED_HEADERS + b"5\r\nhello\r\n0\r\n\r\n"], max_body_bytes=3
    )

    assert outcome.stop is ParseStop.COMPLETE
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"hel"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"5\r\nhel"
    assert leftover == b"lo\r\n0\r\n\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_stop_within_partial_chunk() -> None:
    outcome, leftover = await _parse_body(
        [_CHUNKED_HEADERS + b"5\r\nhel", b"lo\r\n0\r\n\r\n"], max_body_bytes=2
    )

    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"he"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"5\r\nhe"
    assert leftover == b"l"


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_stop_at_chunk_boundary() -> None:
    outcome, leftover = await _parse_body(
        [_CHUNKED_HEADERS + _TWO_CHUNKS], max_body_bytes=3
    )

    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"abc"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"3\r\nabc\r\n"
    assert leftover == b"3\r\ndef\r\n0\r\n\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_stop_across_chunk_boundary() -> (
    None
):
    outcome, leftover = await _parse_body(
        [_CHUNKED_HEADERS + _TWO_CHUNKS], max_body_bytes=4
    )

    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"abcd"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"3\r\nabc\r\n3\r\nd"
    assert leftover == b"ef\r\n0\r\n\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_count_excludes_framing() -> None:
    body = b"3\r\nabc\r\n0\r\n\r\n"

    outcome, leftover = await _parse_body(
        [_CHUNKED_HEADERS + body], max_body_bytes=3
    )

    assert outcome.complete_request is not None
    assert outcome.complete_request.body == b"abc"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + body
    assert leftover == b""


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_spent_budget_stops_next_read() -> (
    None
):
    outcome, leftover = await _parse_body(
        [_CHUNKED_HEADERS, b"3\r\nabc\r\n", b"3\r\ndef\r\n0\r\n\r\n"],
        max_body_bytes=3,
    )

    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"abc"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"3\r\nabc\r\n"
    assert leftover == b"3\r\ndef\r\n0\r\n\r\n"


@pytest.mark.parametrize(
    "script",
    [
        [b"3\r\nabc\r\n3\r\n"],
        [b"3\r\nabc\r\n", b"3\r\n"],
        [b"3\r\nabc\r\n3", b"\r\n"],
    ],
)
@pytest.mark.parametrize(
    "headers", [_CHUNKED_HEADERS, _UPGRADE_CHUNKED_HEADERS]
)
@pytest.mark.asyncio
async def test_continue_parse_body_chunked_spent_budget_stops_at_size_line(
    script: list[bytes],
    headers: bytes,
) -> None:
    outcome, leftover = await _parse_body([headers, *script], max_body_bytes=3)

    assert outcome.stop is ParseStop.COMPLETE
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"abc"
    assert outcome.wire_bytes == headers + b"3\r\nabc\r\n"
    assert leftover == b"3\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_terminal_chunk_completes() -> None:
    outcome, leftover = await _parse_body(
        [_CHUNKED_HEADERS, b"3\r\nabc\r\n", b"0\r\n\r\n"], max_body_bytes=3
    )

    assert outcome.complete_request is not None
    assert outcome.complete_request.body == b"abc"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"3\r\nabc\r\n0\r\n\r\n"
    assert leftover == b""


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_split_reads_stop_at_budget() -> (
    None
):
    script: list[bytes | Exception] = [
        _CHUNKED_HEADERS,
        b"5;ext=1\r\nhel",
        b"lo\r\n3\r\nabc\r\n0\r\nX-Trailer: v\r\n\r\n",
    ]

    outcome, leftover = await _parse_body(script, max_body_bytes=6)

    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"helloa"
    assert outcome.wire_bytes == (
        _CHUNKED_HEADERS + b"5;ext=1\r\nhello\r\n3\r\na"
    )
    assert leftover == b"bc\r\n0\r\nX-Trailer: v\r\n\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_exact_budget_completes() -> None:
    chunked_body = b"5;ext=1\r\nhello\r\n3\r\nabc\r\n0\r\nX-Trailer: v\r\n\r\n"
    script: list[bytes | Exception] = [
        _CHUNKED_HEADERS,
        chunked_body[:12],
        chunked_body[12:],
    ]

    outcome, leftover = await _parse_body(script, max_body_bytes=8)

    assert outcome.complete_request is not None
    assert outcome.complete_request.body == b"helloabc"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + chunked_body
    assert leftover == b""


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_eof_mid_chunk_stops_with_eof() -> (
    None
):
    outcome, leftover = await _parse_body([_CHUNKED_HEADERS + b"5\r\nhel"])

    assert outcome.stop is ParseStop.EOF
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"hel"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"5\r\nhel"
    assert leftover == b""


@pytest.mark.asyncio
async def test_continue_parse_body_chunked_read_error_keeps_chunks() -> None:
    error = ConnectionResetError("Connection reset")

    outcome, _ = await _parse_body([
        _CHUNKED_HEADERS + b"5\r\nhello\r\n",
        error,
    ])

    assert outcome.stop is ParseStop.READ_ERROR
    assert outcome.error is error
    assert _partial(outcome).body == b"hello"
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"5\r\nhello\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_upgrade_chunked_stop_within_chunk() -> None:
    outcome, leftover = await _parse_body(
        [_UPGRADE_CHUNKED_HEADERS + b"5\r\nhello\r\n0\r\n\r\n"],
        max_body_bytes=3,
    )

    assert outcome.stop is ParseStop.COMPLETE
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"hel"
    assert outcome.wire_bytes == _UPGRADE_CHUNKED_HEADERS + b"5\r\nhel"
    assert leftover == b"lo\r\n0\r\n\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_upgrade_chunked_eof_keeps_partial() -> None:
    outcome, _ = await _parse_body([_UPGRADE_CHUNKED_HEADERS + b"5\r\nhel"])

    assert outcome.stop is ParseStop.EOF
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"hel"


@pytest.mark.parametrize(
    "headers", [_CHUNKED_HEADERS, _UPGRADE_CHUNKED_HEADERS]
)
@pytest.mark.asyncio
async def test_continue_parse_body_chunked_eof_after_size_line_keeps_parts(
    headers: bytes,
) -> None:
    outcome, leftover = await _parse_body([headers + b"3\r\nabc\r\n3\r\n"])

    assert outcome.stop is ParseStop.EOF
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body_parts == [b"abc"]
    assert outcome.wire_bytes == headers + b"3\r\nabc\r\n3\r\n"
    assert leftover == b""


@pytest.mark.asyncio
async def test_continue_parse_body_upgrade_bad_chunk_records_prefix() -> None:
    outcome, leftover = await _parse_body([
        _UPGRADE_CHUNKED_HEADERS + b"Z\r\n"
    ])

    assert outcome.stop is ParseStop.PARSE_ERROR
    assert isinstance(outcome.error, ChunkScanError)
    assert _partial(outcome).body == b""
    assert outcome.wire_bytes == _UPGRADE_CHUNKED_HEADERS + b"Z"
    assert leftover == b"\r\n"


@pytest.mark.parametrize(
    ("script", "prefix"),
    [
        ([b"3\r\nabc\r\nZ\r\n"], b"3\r\nabc\r\nZ"),
        ([b"3\r\nabc\r\n", b"Z\r\n"], b"3\r\nabc\r\nZ"),
        ([b"3\r\nabcX\r\n"], b"3\r\nabcX"),
    ],
)
@pytest.mark.parametrize(
    "headers", [_CHUNKED_HEADERS, _UPGRADE_CHUNKED_HEADERS]
)
@pytest.mark.asyncio
async def test_continue_parse_body_bad_chunk_keeps_payload_prefix(
    script: list[bytes | Exception],
    prefix: bytes,
    headers: bytes,
) -> None:
    outcome, leftover = await _parse_body([headers, *script])

    assert outcome.stop is ParseStop.PARSE_ERROR
    assert isinstance(outcome.error, ChunkScanError)
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body_parts == [b"abc"]
    assert outcome.wire_bytes == headers + prefix
    assert leftover == b"\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_upgrade_content_length_stops_early() -> (
    None
):
    headers = _content_length_headers(5, upgrade=True)

    outcome, leftover = await _parse_body(
        [headers + b"hello"], max_body_bytes=2
    )

    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b"he"
    assert outcome.wire_bytes == headers + b"he"
    assert leftover == b"llo"


@pytest.mark.asyncio
async def test_continue_parse_body_upgrade_content_length_completes() -> None:
    headers = _content_length_headers(5, upgrade=True)

    outcome, leftover = await _parse_body(
        [headers + b"hello"], max_body_bytes=5
    )

    assert outcome.complete_request is not None
    assert outcome.complete_request.body == b"hello"
    assert leftover == b""


@pytest.mark.parametrize("max_body_bytes", [None, 0, 5])
@pytest.mark.asyncio
async def test_continue_parse_body_upgrade_zero_content_length_completes(
    max_body_bytes: int | None,
) -> None:
    headers = _content_length_headers(0, upgrade=True)

    outcome, leftover = await _parse_body(
        [headers + b"tail"], max_body_bytes=max_body_bytes
    )

    assert outcome.complete_request is not None
    assert outcome.complete_request.body == b""
    assert outcome.wire_bytes == headers
    assert leftover == b"tail"


@pytest.mark.asyncio
async def test_continue_parse_body_bad_extension_in_chunk_is_parse_error() -> (
    None
):
    outcome, leftover = await _parse_body([
        _CHUNKED_HEADERS,
        _BAD_EXTENSION_CHUNK,
        b"0\r\n\r\n",
    ])

    assert outcome.stop is ParseStop.PARSE_ERROR
    assert isinstance(outcome.error, httptools.HttpParserError)
    assert _partial(outcome).body == b""
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"5;a\x01"
    assert leftover == b"\r\nhello\r\n"


@pytest.mark.asyncio
async def test_continue_parse_body_bad_extension_at_end_is_parse_error() -> (
    None
):
    outcome, leftover = await _parse_body([
        _CHUNKED_HEADERS + _BAD_EXTENSION_CHUNK + b"0\r\n\r\n",
    ])

    assert outcome.stop is ParseStop.PARSE_ERROR
    assert isinstance(outcome.error, httptools.HttpParserError)
    assert _partial(outcome).body == b""
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"5;a\x01"
    assert leftover == b"\r\nhello\r\n0\r\n\r\n"


def test_snapshot_before_parsing_reports_no_request() -> None:
    outcome = AsyncRequestParser().snapshot()

    assert outcome.stop is ParseStop.COMPLETE
    assert outcome.parsed is None
    assert outcome.wire_bytes == b""
    assert outcome.error is None


@pytest.mark.asyncio
async def test_snapshot_after_headers_reports_incomplete_request() -> None:
    headers = _content_length_headers(5)
    reader = _ScriptedReader([headers + b"hello"])
    parser = AsyncRequestParser()
    await parser.parse_headers(reader)

    outcome = parser.snapshot()

    assert outcome.stop is ParseStop.COMPLETE
    assert _partial(outcome).headers_complete
    assert not _partial(outcome).is_complete
    assert _partial(outcome).body == b""
    assert outcome.wire_bytes == headers
    assert outcome.complete_request is None


@pytest.mark.asyncio
async def test_snapshot_after_cancelled_body_read_keeps_delivered_chunks() -> (
    None
):
    reader = asyncio.StreamReader()
    reader.feed_data(_CHUNKED_HEADERS + b"5\r\nhello\r\n")
    parser = AsyncRequestParser()
    task = asyncio.create_task(parser.parse(reader))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    outcome = parser.snapshot()

    assert outcome.stop is ParseStop.COMPLETE
    assert _partial(outcome).body == b"hello"
    assert not _partial(outcome).is_complete
    assert outcome.wire_bytes == _CHUNKED_HEADERS + b"5\r\nhello\r\n"
    assert outcome.complete_request is None


@pytest.mark.asyncio
async def test_http_request_reader_returns_none_for_truncated_body() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(_content_length_headers(10) + b"short")
    reader.feed_eof()

    assert await HTTPRequestReader().read_request(reader) is None


@pytest.mark.asyncio
async def test_http_request_reader_returns_none_on_read_error() -> None:
    reader = _ScriptedReader([ConnectionResetError("Connection reset")])

    assert await HTTPRequestReader().read_request(reader) is None


@pytest.mark.asyncio
async def test_http_request_reader_returns_none_for_bad_chunk_framing() -> (
    None
):
    reader = asyncio.StreamReader()
    reader.feed_data(_CHUNKED_HEADERS + b"5\r\nhello\r\nZ\r\n")
    reader.feed_eof()

    assert await HTTPRequestReader().read_request(reader) is None


@pytest.mark.asyncio
async def test_http_request_reader_content_length_zero_has_empty_body() -> (
    None
):
    reader = asyncio.StreamReader()
    reader.feed_data(_content_length_headers(0))
    reader.feed_eof()

    request = await HTTPRequestReader().read_request(reader)

    assert request is not None
    assert request.body == b""
    assert request.body_complete


@pytest.mark.asyncio
async def test_http_request_reader_empty_chunked_body_is_empty_bytes() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(_CHUNKED_HEADERS + b"0\r\n\r\n")
    reader.feed_eof()

    request = await HTTPRequestReader().read_request(reader)

    assert request is not None
    assert request.body == b""
    assert request.body_complete


@pytest.mark.asyncio
async def test_http_request_reader_without_content_headers_has_no_body() -> (
    None
):
    reader = asyncio.StreamReader()
    reader.feed_data(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
    reader.feed_eof()

    request = await HTTPRequestReader().read_request(reader)

    assert request is not None
    assert request.body is None
    assert request.body_complete


@pytest.mark.asyncio
async def test_from_parsed_threshold_outcome_marks_body_incomplete() -> None:
    headers = _content_length_headers(5)
    outcome, _ = await _parse_body([headers + b"hello"], max_body_bytes=2)

    recorded = RecordedHTTPRequest.from_parsed(
        _partial(outcome), outcome.wire_bytes
    )

    assert not recorded.body_complete
    assert recorded.body == b"he"
    assert recorded.wire_body_bytes == b"he"


@pytest.mark.asyncio
@given(data=st.data(), chunked=st.booleans())
async def test_body_cutoffs_preserve_payload_prefix_and_remaining_stream(
    data: st.DataObject,
    chunked: bool,
) -> None:
    if chunked:
        body = data.draw(chunked_bodies())
        payload = b"".join(body.payloads)
        encoded = _CHUNKED_HEADERS + body.encoded
        boundaries = list(itertools.accumulate(map(len, body.payloads)))
    else:
        payload = data.draw(st.binary(max_size=128))
        encoded = _content_length_headers(len(payload)) + payload
        boundaries = [len(payload)]
    cutoff = data.draw(
        st.one_of(
            st.sampled_from([0, *boundaries, len(payload), len(payload) + 1]),
            st.integers(min_value=0, max_value=len(payload) + 1),
        )
    )
    original = encoded + _NEXT_REQUEST
    fragments = data.draw(fragments_of(original))
    # Also force every CRLF (and every payload boundary) across reads.
    for parts in (fragments, [bytes([byte]) for byte in original]):
        reader = _ScriptedReader(list(parts))
        parser = AsyncRequestParser()
        staged, remaining = await parser.parse_headers(reader)
        assert staged.parsed is not None
        outcome = await parser.continue_parse_body(
            reader,
            remaining,
            max_body_bytes=cutoff,
        )
        parsed = _partial(outcome)
        recorded = RecordedHTTPRequest.from_parsed(parsed, outcome.wire_bytes)
        rest = take_unread_data(reader)
        while piece := await reader.read():
            rest += piece

        assert recorded.body == payload[:cutoff]
        assert outcome.wire_bytes + rest == original
        assert rest.endswith(_NEXT_REQUEST)
        assert recorded.body_complete == (outcome.wire_bytes == encoded)
        # At an exact chunk cutoff, available framing determines whether
        # parsing reaches the message boundary before stopping.
        if not chunked or cutoff != len(payload):
            assert recorded.body_complete == (cutoff >= len(payload))
        if recorded.body_complete:
            assert rest == _NEXT_REQUEST
