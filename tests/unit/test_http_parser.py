"""Unit tests for the http_parser module."""

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
    RequestProtocol,
)
from localstub.http.response import (
    AsyncResponseParser,
    ParsedResponse,
    ResponseProtocol,
)
from localstub.http.utils import headers_to_message


class TestParsedRequest:
    """Tests for ParsedRequest dataclass."""

    def test_default_values(self):
        """Test default values for ParsedRequest."""
        req = ParsedRequest()
        assert req.method is None
        assert req.url is None
        assert req.http_version is None
        assert req.headers == []
        assert req.body_parts == []
        assert not req.is_complete
        assert not req.headers_complete

    def test_body_property_empty(self):
        """Test body property with empty body_parts."""
        req = ParsedRequest()
        assert req.body == b""

    def test_body_property_single_part(self):
        """Test body property with single body part."""
        req = ParsedRequest(body_parts=[b"hello"])
        assert req.body == b"hello"

    def test_body_property_multiple_parts(self):
        """Test body property with multiple body parts."""
        req = ParsedRequest(body_parts=[b"hello", b" ", b"world"])
        assert req.body == b"hello world"


class TestParsedResponse:
    """Tests for ParsedResponse dataclass."""

    def test_default_values(self):
        """Test default values for ParsedResponse."""
        resp = ParsedResponse()
        assert resp.status_code is None
        assert resp.status_text is None
        assert resp.http_version is None
        assert resp.headers == []
        assert resp.body_parts == []
        assert resp.is_complete is False

    def test_body_property_empty(self):
        """Test body property with empty body_parts."""
        resp = ParsedResponse()
        assert resp.body == b""

    def test_body_property_single_part(self):
        """Test body property with single body part."""
        resp = ParsedResponse(body_parts=[b"response body"])
        assert resp.body == b"response body"

    def test_body_property_multiple_parts(self):
        """Test body property with multiple body parts."""
        resp = ParsedResponse(body_parts=[b"part1", b"part2", b"part3"])
        assert resp.body == b"part1part2part3"


class TestRequestProtocol:
    """Tests for RequestProtocol callback handler."""

    def test_on_message_begin_resets_result(self):
        """Test on_message_begin resets the result."""
        protocol = RequestProtocol()
        protocol.result.method = "GET"
        protocol.result.url = b"/test"

        protocol.on_message_begin()

        assert protocol.result.method is None
        assert protocol.result.url is None

    def test_on_url(self):
        """Test on_url sets the URL."""
        protocol = RequestProtocol()
        protocol.on_url(b"/api/users")
        assert protocol.result.url == b"/api/users"

    def test_on_header(self):
        """Test on_header adds headers."""
        protocol = RequestProtocol()
        protocol.on_header(b"Content-Type", b"application/json")
        protocol.on_header(b"Host", b"localhost")

        assert len(protocol.result.headers) == 2
        assert protocol.result.headers[0] == (
            b"Content-Type",
            b"application/json",
        )
        assert protocol.result.headers[1] == (b"Host", b"localhost")

    def test_on_body(self):
        """Test on_body adds body parts."""
        protocol = RequestProtocol()
        protocol.on_body(b"chunk1")
        protocol.on_body(b"chunk2")

        assert protocol.result.body_parts == [b"chunk1", b"chunk2"]

    def test_on_message_complete(self):
        """Test on_message_complete sets is_complete."""
        protocol = RequestProtocol()
        assert protocol.result.is_complete is False

        protocol.on_message_complete()

        assert protocol.result.is_complete is True

    def test_on_chunk_header_and_complete_are_noop(self):
        """Test chunk callbacks are no-op."""
        protocol = RequestProtocol()
        # These should not raise
        protocol.on_chunk_header()
        protocol.on_chunk_complete()

    def test_callbacks_are_noop_after_message_complete(self):
        """Test that callbacks are no-ops after message is complete.

        This protects against pipelined requests overwriting the first
        completed request when httptools processes multiple requests
        in a single buffer.
        """
        protocol = RequestProtocol()

        # Simulate first request
        protocol.on_message_begin()
        protocol.on_url(b"/first")
        protocol.on_header(b"Host", b"first.com")
        protocol.on_body(b"first body")
        protocol.on_message_complete()

        # Verify first request is complete
        assert protocol.result.is_complete is True
        assert protocol.result.url == b"/first"
        assert protocol.result.headers == [(b"Host", b"first.com")]
        assert protocol.result.body_parts == [b"first body"]

        # Simulate second pipelined request callbacks - should be ignored
        protocol.on_message_begin()
        protocol.on_url(b"/second")
        protocol.on_header(b"Host", b"second.com")
        protocol.on_body(b"second body")
        protocol.on_message_complete()

        # First request should be preserved
        assert protocol.result.url == b"/first"
        assert protocol.result.headers == [(b"Host", b"first.com")]
        assert protocol.result.body_parts == [b"first body"]

    def test_on_headers_complete_without_parser_is_noop(self):
        protocol = RequestProtocol()
        protocol.result.method = "GET"
        protocol.result.url = b"/test"

        protocol.on_headers_complete()

        assert protocol.result.method == "GET"
        assert protocol.result.url == b"/test"
        assert protocol.result.http_version is None

    def test_on_url_accumulates_chunks(self):
        protocol = RequestProtocol()

        protocol.on_url(b"/api")
        protocol.on_url(b"/users")
        protocol.on_url(b"/123")

        assert protocol.result.url == b"/api/users/123"


class TestResponseProtocol:
    """Tests for ResponseProtocol callback handler."""

    def test_on_message_begin_resets_result(self):
        """Test on_message_begin resets the result."""
        protocol = ResponseProtocol()
        protocol.result.status_code = 200
        protocol.result.status_text = b"OK"

        protocol.on_message_begin()

        assert protocol.result.status_code is None
        assert protocol.result.status_text is None

    def test_on_status(self):
        """Test on_status sets the status text."""
        protocol = ResponseProtocol()
        protocol.on_status(b"OK")
        assert protocol.result.status_text == b"OK"

    def test_on_header(self):
        """Test on_header adds headers."""
        protocol = ResponseProtocol()
        protocol.on_header(b"Content-Type", b"text/html")
        protocol.on_header(b"Content-Length", b"100")

        assert len(protocol.result.headers) == 2
        assert protocol.result.headers[0] == (b"Content-Type", b"text/html")
        assert protocol.result.headers[1] == (b"Content-Length", b"100")

    def test_on_body(self):
        """Test on_body adds body parts."""
        protocol = ResponseProtocol()
        protocol.on_body(b"response data")

        assert protocol.result.body_parts == [b"response data"]

    def test_on_message_complete(self):
        """Test on_message_complete sets is_complete."""
        protocol = ResponseProtocol()
        assert protocol.result.is_complete is False

        protocol.on_message_complete()

        assert protocol.result.is_complete is True

    def test_on_chunk_header_and_complete_are_noop(self):
        """Test chunk callbacks are no-op."""
        protocol = ResponseProtocol()
        # These should not raise
        protocol.on_chunk_header()
        protocol.on_chunk_complete()

    def test_callbacks_are_noop_after_message_complete(self):
        """Test that callbacks are no-ops after message is complete.

        This protects against pipelined responses overwriting the first
        completed response when httptools processes multiple responses
        in a single buffer.
        """
        protocol = ResponseProtocol()

        # Simulate first response
        protocol.on_message_begin()
        protocol.on_status(b"OK")
        protocol.on_header(b"Content-Type", b"text/html")
        protocol.on_body(b"first body")
        protocol.on_message_complete()

        # Verify first response is complete
        assert protocol.result.is_complete is True
        assert protocol.result.status_text == b"OK"
        assert protocol.result.headers == [(b"Content-Type", b"text/html")]
        assert protocol.result.body_parts == [b"first body"]

        # Simulate second pipelined response callbacks - should be ignored
        protocol.on_message_begin()
        protocol.on_status(b"Not Found")
        protocol.on_header(b"Content-Type", b"text/plain")
        protocol.on_body(b"second body")
        protocol.on_message_complete()

        # First response should be preserved
        assert protocol.result.status_text == b"OK"
        assert protocol.result.headers == [(b"Content-Type", b"text/html")]
        assert protocol.result.body_parts == [b"first body"]

    def test_on_status_accumulates_chunks(self):
        protocol = ResponseProtocol()

        protocol.on_status(b"Not")
        protocol.on_status(b" ")
        protocol.on_status(b"Found")

        assert protocol.result.status_text == b"Not Found"

    def test_on_headers_complete_without_parser_is_noop(self):
        protocol = ResponseProtocol()
        protocol.result.status_text = b"OK"

        protocol.on_headers_complete()

        assert protocol.result.status_text == b"OK"
        assert protocol.result.status_code is None
        assert protocol.result.http_version is None


class TestHeadersToMessage:
    """Tests for headers_to_message function."""

    def test_empty_headers(self):
        """Test with empty headers list."""
        msg = headers_to_message([])
        assert len(msg.keys()) == 0

    def test_single_header(self):
        """Test with a single header."""
        msg = headers_to_message([(b"Content-Type", b"application/json")])
        assert msg["Content-Type"] == "application/json"

    def test_multiple_headers(self):
        """Test with multiple headers."""
        headers = [
            (b"Host", b"example.com"),
            (b"Accept", b"*/*"),
            (b"Connection", b"keep-alive"),
        ]
        msg = headers_to_message(headers)
        assert msg["Host"] == "example.com"
        assert msg["Accept"] == "*/*"
        assert msg["Connection"] == "keep-alive"

    def test_iso8859_encoding(self):
        """Test headers are decoded as ISO-8859-1."""
        # ISO-8859-1 allows bytes 0x80-0xFF
        msg = headers_to_message([(b"X-Custom", b"\xe4\xf6\xfc")])
        assert msg["X-Custom"] == "\xe4\xf6\xfc"


class TestAsyncRequestParser:
    """Tests for AsyncRequestParser."""

    @pytest.mark.asyncio
    async def test_parse_simple_get_request(self):
        """Test parsing a simple GET request."""
        request_data = b"GET /api/test HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = _create_mock_reader([request_data])

        parser = AsyncRequestParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is not None
        assert parsed.method == "GET"
        assert parsed.url == b"/api/test"
        assert parsed.http_version == "1.1"
        assert parsed.is_complete is True
        assert wire_bytes == request_data

    @pytest.mark.asyncio
    async def test_parse_post_request_with_body(self):
        """Test parsing a POST request with Content-Length body."""
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
    async def test_parse_chunked_request(self):
        """Test parsing a chunked transfer-encoded request."""
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
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is not None
        assert parsed.method == "POST"
        assert parsed.body == b"Hello World"
        assert parsed.is_complete is True

    @pytest.mark.asyncio
    async def test_parse_with_connection_wire(self):
        """Test that connection_wire is populated."""
        request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = _create_mock_reader([request_data])
        connection_wire = bytearray()

        parser = AsyncRequestParser()
        await parser.parse(reader, connection_wire)

        assert bytes(connection_wire) == request_data

    @pytest.mark.asyncio
    async def test_parse_eof_before_complete(self):
        """Test handling EOF before message is complete."""
        # Incomplete request - missing headers end
        reader = _create_mock_reader([b"GET / HTTP/1.1\r\nHost:"])

        parser = AsyncRequestParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None

    @pytest.mark.asyncio
    async def test_parse_malformed_request(self):
        """Test handling malformed request."""
        # Completely invalid HTTP
        reader = _create_mock_reader([b"NOT VALID HTTP AT ALL"])

        parser = AsyncRequestParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None

    @pytest.mark.asyncio
    async def test_wire_bytes_property(self):
        """Test wire_bytes property."""
        request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = _create_mock_reader([request_data])

        parser = AsyncRequestParser()
        await parser.parse(reader)

        assert parser.wire_bytes == request_data

    @pytest.mark.asyncio
    async def test_parse_pipelined_requests_preserves_first(self):
        """Test that pipelined requests preserve the first request.

        When two HTTP requests arrive in the same TCP frame, the parser
        must preserve the first completed request and push leftover bytes
        back to the reader for the next request.
        """
        first_request = b"GET /first HTTP/1.1\r\nHost: first.com\r\n\r\n"
        second_request = b"GET /second HTTP/1.1\r\nHost: second.com\r\n\r\n"
        pipelined_data = first_request + second_request

        # Use a real StreamReader to test feed_data behavior
        # Don't call feed_eof() yet - we need to push leftover bytes back
        reader = asyncio.StreamReader()
        reader.feed_data(pipelined_data)

        # Parse first request
        parser1 = AsyncRequestParser()
        parsed1, wire_bytes1 = await parser1.parse(reader)

        # First request should be returned correctly
        assert parsed1 is not None
        assert parsed1.method == "GET"
        assert parsed1.url == b"/first"
        assert parsed1.is_complete is True
        host_value = next(h[1] for h in parsed1.headers if h[0] == b"Host")
        assert host_value == b"first.com"
        # Wire bytes should only contain the first request
        assert wire_bytes1 == first_request

        # Now signal EOF for the second request
        reader.feed_eof()

        # Parse second request from the same reader
        # The leftover bytes should have been pushed back
        parser2 = AsyncRequestParser()
        parsed2, wire_bytes2 = await parser2.parse(reader)

        # Second request should also be parsed correctly
        assert parsed2 is not None
        assert parsed2.method == "GET"
        assert parsed2.url == b"/second"
        assert parsed2.is_complete is True
        host_value2 = next(h[1] for h in parsed2.headers if h[0] == b"Host")
        assert host_value2 == b"second.com"
        assert wire_bytes2 == second_request

    @pytest.mark.asyncio
    async def test_parse_with_read_exception_returns_none(self):
        reader = AsyncMock(spec=asyncio.StreamReader)
        reader.read = AsyncMock(
            side_effect=ConnectionResetError("Connection reset")
        )

        parser = AsyncRequestParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None
        assert wire_bytes == b""

    @pytest.mark.asyncio
    async def test_parse_malformed_with_remaining_buffer_pushes_back(self):
        reader = asyncio.StreamReader()
        # Feed malformed HTTP followed by what looks like more data
        # The parser will fail on the malformed part and should push back
        # remaining buffer bytes. Don't call feed_eof() so push back works.
        malformed_with_extra = b"INVALID HTTP\x00\x01\x02extra data here"
        reader.feed_data(malformed_with_extra)

        parser = AsyncRequestParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None
        assert len(wire_bytes) > 0
        # Verify remaining bytes were pushed back to the reader
        remaining = await reader.read(100)
        assert len(remaining) > 0

    @pytest.mark.asyncio
    async def test_parse_headers_stops_after_headers(self):
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
        parsed, header_wire, remaining = await parser.parse_headers(reader)

        assert parsed is not None
        assert parsed.method == "POST"
        assert parsed.url == b"/upload"
        assert parsed.headers_complete
        # Body should not be parsed yet
        assert parsed.body == b""
        assert not parsed.is_complete
        # Header wire bytes should not include body
        assert b"hello" not in header_wire
        assert b"\r\n\r\n" in header_wire

    @pytest.mark.asyncio
    async def test_parse_headers_sets_headers_complete_flag(self):
        request_data = b"GET /test HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = asyncio.StreamReader()
        reader.feed_data(request_data)
        reader.feed_eof()

        parser = AsyncRequestParser()
        parsed, _, _ = await parser.parse_headers(reader)

        assert parsed is not None
        assert parsed.headers_complete
        # For a GET with no body, headers_complete and is_complete may both
        # be true after headers are parsed
        assert parsed.method == "GET"

    @pytest.mark.asyncio
    async def test_parse_headers_returns_remaining_buffer(self):
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
        parsed, header_wire, remaining = await parser.parse_headers(reader)

        assert parsed is not None
        # The remaining buffer should contain the body bytes
        # (either in remaining or pushed back to reader)
        # Since we read byte-by-byte, remaining may contain body
        assert len(remaining) > 0 or not reader.at_eof()

    @pytest.mark.asyncio
    async def test_continue_parse_body_completes_request(self):
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
        # First parse headers
        parsed, header_wire, remaining = await parser.parse_headers(reader)
        assert parsed is not None
        assert not parsed.is_complete

        # Then continue parsing body
        parsed, wire_bytes = await parser.continue_parse_body(
            reader, remaining
        )

        assert parsed is not None
        assert parsed.is_complete
        assert parsed.body == b"hello"
        # Complete wire bytes should include everything
        assert wire_bytes == request_data

    @pytest.mark.asyncio
    async def test_parse_headers_with_connection_wire(self):
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
        parsed, header_wire, remaining = await parser.parse_headers(
            reader, connection_wire
        )

        assert parsed is not None
        # Connection wire should have accumulated header bytes
        assert len(connection_wire) > 0
        assert b"POST /upload" in bytes(connection_wire)

    @pytest.mark.asyncio
    async def test_continue_parse_body_with_connection_wire(self):
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
        parsed, header_wire, remaining = await parser.parse_headers(
            reader, connection_wire
        )

        # Continue parsing body with same connection_wire
        parsed, wire_bytes = await parser.continue_parse_body(
            reader, remaining, connection_wire
        )

        assert parsed is not None
        assert parsed.is_complete
        # Connection wire should have all bytes
        assert bytes(connection_wire) == request_data

    @pytest.mark.asyncio
    async def test_parse_headers_eof_before_complete(self):
        # Incomplete headers
        reader = asyncio.StreamReader()
        reader.feed_data(b"GET / HTTP/1.1\r\nHost:")
        reader.feed_eof()

        parser = AsyncRequestParser()
        parsed, header_wire, remaining = await parser.parse_headers(reader)

        assert parsed is None

    @pytest.mark.asyncio
    async def test_parse_headers_malformed_request(self):
        reader = asyncio.StreamReader()
        reader.feed_data(b"INVALID HTTP DATA")

        parser = AsyncRequestParser()
        parsed, header_wire, remaining = await parser.parse_headers(reader)

        assert parsed is None

    @pytest.mark.asyncio
    async def test_continue_parse_body_eof_before_complete(self):
        # Headers complete but body truncated
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
        parsed, header_wire, remaining = await parser.parse_headers(reader)
        assert parsed is not None

        parsed, wire_bytes = await parser.continue_parse_body(
            reader, remaining
        )

        assert parsed is None


class TestAsyncResponseParser:
    """Tests for AsyncResponseParser."""

    @pytest.mark.asyncio
    async def test_parse_simple_response(self):
        """Test parsing a simple HTTP response."""
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
    async def test_parse_chunked_response(self):
        """Test parsing a chunked transfer-encoded response."""
        response_data = (
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"5\r\nHello\r\n"
            b"0\r\n\r\n"
        )
        reader = _create_mock_reader([response_data])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is not None
        assert parsed.status_code == 200
        assert parsed.body == b"Hello"

    @pytest.mark.asyncio
    async def test_parse_close_delimited_response(self):
        """Test parsing a close-delimited response (no Content-Length)."""
        # Response without Content-Length, body ends when connection closes
        response_data = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n"
            b"Response body without length"
        )
        reader = _create_mock_reader([response_data])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is not None
        assert parsed.status_code == 200
        assert parsed.body == b"Response body without length"

    @pytest.mark.asyncio
    async def test_parse_empty_response_body(self):
        """Test parsing response with empty body."""
        response_data = b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n"
        reader = _create_mock_reader([response_data])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is not None
        assert parsed.status_code == 204
        assert parsed.body == b""

    @pytest.mark.asyncio
    async def test_parse_eof_before_headers(self):
        """Test handling EOF before headers are complete."""
        reader = _create_mock_reader([b"HTTP/1.1 200"])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None

    @pytest.mark.asyncio
    async def test_wire_bytes_property(self):
        """Test wire_bytes property."""
        response_data = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"
        reader = _create_mock_reader([response_data])

        parser = AsyncResponseParser()
        await parser.parse(reader)

        assert parser.wire_bytes == response_data

    @pytest.mark.asyncio
    async def test_parse_with_read_exception_returns_none(self):
        reader = AsyncMock(spec=asyncio.StreamReader)
        reader.read = AsyncMock(side_effect=OSError("Connection lost"))

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None
        assert wire_bytes == b""

    @pytest.mark.asyncio
    async def test_parse_head_response_completes_after_headers(self):
        response_data = b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\n"
        reader = _create_mock_reader([response_data])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader, request_method="HEAD")

        assert parsed is not None
        assert parsed.status_code == 200
        assert parsed.is_complete
        assert parsed.body == b""

    @pytest.mark.asyncio
    async def test_parse_malformed_response_returns_none(self):
        reader = _create_mock_reader([b"NOT A VALID HTTP RESPONSE\r\n"])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None

    @pytest.mark.asyncio
    async def test_parse_truncated_content_length_response_returns_none(self):
        # Response declares 100 bytes but only 10 are provided before EOF
        response_data = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/plain\r\n"
            b"Content-Length: 100\r\n"
            b"\r\n"
            b"Truncated!"
        )
        reader = _create_mock_reader([response_data])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None

    @pytest.mark.asyncio
    async def test_parse_truncated_chunked_response_returns_none(self):
        # Chunked response without the terminating "0\r\n\r\n"
        response_data = (
            b"HTTP/1.1 200 OK\r\n"
            b"Transfer-Encoding: chunked\r\n"
            b"\r\n"
            b"5\r\nHello\r\n"
        )
        reader = _create_mock_reader([response_data])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None

    @pytest.mark.asyncio
    async def test_parse_truncated_lowercase_content_length_returns_none(self):
        # Verify case-insensitive header detection for content-length
        response_data = (
            b"HTTP/1.1 200 OK\r\ncontent-length: 100\r\n\r\npartial"
        )
        reader = _create_mock_reader([response_data])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None

    @pytest.mark.asyncio
    async def test_parse_truncated_uppercase_transfer_encoding_returns_none(
        self,
    ):
        # Verify case-insensitive header detection for TRANSFER-ENCODING
        response_data = (
            b"HTTP/1.1 200 OK\r\n"
            b"TRANSFER-ENCODING: chunked\r\n"
            b"\r\n"
            b"5\r\nHello\r\n"
        )
        reader = _create_mock_reader([response_data])

        parser = AsyncResponseParser()
        parsed, wire_bytes = await parser.parse(reader)

        assert parsed is None


class TestHTTPRequestReader:
    @pytest.mark.asyncio
    async def test_read_request_simple_get(self):
        request_data = b"GET /api/test HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = asyncio.StreamReader()
        reader.feed_data(request_data)
        reader.feed_eof()

        http_reader = HTTPRequestReader()
        request = await http_reader.read_request(reader)

        assert request is not None
        assert request.method == "GET"
        assert request.path == "/api/test"
        assert request.http_version == "1.1"
        assert request.wire_raw_bytes == request_data

    @pytest.mark.asyncio
    async def test_read_request_with_body(self):
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
        assert request.body == "Hello, World!"

    @pytest.mark.asyncio
    async def test_read_request_returns_none_on_parse_failure(self):
        reader = asyncio.StreamReader()
        # Don't call feed_eof() - the malformed request triggers parse error
        # and the code tries to push back remaining bytes
        reader.feed_data(b"NOT VALID HTTP")

        http_reader = HTTPRequestReader()
        request = await http_reader.read_request(reader)

        assert request is None

    @pytest.mark.asyncio
    async def test_read_request_returns_none_on_eof(self):
        reader = asyncio.StreamReader()
        reader.feed_eof()

        http_reader = HTTPRequestReader()
        request = await http_reader.read_request(reader)

        assert request is None

    @pytest.mark.asyncio
    async def test_read_request_with_connection_wire(self):
        request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = asyncio.StreamReader()
        reader.feed_data(request_data)
        reader.feed_eof()
        connection_wire = bytearray()

        http_reader = HTTPRequestReader()
        request = await http_reader.read_request(
            reader, connection_wire=connection_wire
        )

        assert request is not None
        assert bytes(connection_wire) == request_data

    @pytest.mark.asyncio
    async def test_read_request_extracts_client_info_from_writer(self):
        request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = asyncio.StreamReader()
        reader.feed_data(request_data)
        reader.feed_eof()

        writer = Mock()
        writer.get_extra_info = Mock(return_value=("127.0.0.1", 54321))

        http_reader = HTTPRequestReader()
        request = await http_reader.read_request(reader, writer=writer)

        assert request is not None
        assert request.client == ("127.0.0.1", 54321)
        writer.get_extra_info.assert_called_once_with("peername")

    @pytest.mark.asyncio
    async def test_read_request_with_invalid_peer_returns_none_client(self):
        request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = asyncio.StreamReader()
        reader.feed_data(request_data)
        reader.feed_eof()

        writer = Mock()
        writer.get_extra_info = Mock(return_value=None)

        http_reader = HTTPRequestReader()
        request = await http_reader.read_request(reader, writer=writer)

        assert request is not None
        assert request.client is None

    @pytest.mark.asyncio
    async def test_read_request_with_short_peer_tuple_returns_none_client(
        self,
    ):
        request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = asyncio.StreamReader()
        reader.feed_data(request_data)
        reader.feed_eof()

        writer = Mock()
        writer.get_extra_info = Mock(return_value=("127.0.0.1",))

        http_reader = HTTPRequestReader()
        request = await http_reader.read_request(reader, writer=writer)

        assert request is not None
        assert request.client is None

    @pytest.mark.asyncio
    async def test_read_request_without_writer_has_none_client(self):
        request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = asyncio.StreamReader()
        reader.feed_data(request_data)
        reader.feed_eof()

        http_reader = HTTPRequestReader()
        request = await http_reader.read_request(reader, writer=None)

        assert request is not None
        assert request.client is None

    @pytest.mark.asyncio
    async def test_read_request_with_custom_max_read(self):
        request_data = b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n"
        reader = asyncio.StreamReader()
        reader.feed_data(request_data)
        reader.feed_eof()

        http_reader = HTTPRequestReader(max_read=1024)
        request = await http_reader.read_request(reader)

        assert request is not None
        assert request.method == "GET"


class TestHTTPRequest:
    def test_json_body_returns_none_for_empty_body(self):
        request = HTTPRequest(method="GET", path="/", http_version="1.1")
        assert request.json_body is None

    def test_json_body_returns_none_for_empty_string(self):
        request = HTTPRequest(
            method="GET",
            path="/",
            http_version="1.1",
            body="",
        )
        assert request.json_body is None

    def test_json_body_parses_json(self):
        request = HTTPRequest(
            method="GET",
            path="/",
            http_version="1.1",
            body='{"key": "value"}',
        )
        assert request.json_body == {"key": "value"}

    def test_headers_are_immutable(self) -> None:
        headers = Headers.from_items([("X-Test", "a")])
        request = HTTPRequest(
            method="GET",
            path="/",
            http_version="1.1",
            headers=headers,
        )

        assert request.headers["X-Test"] == "a"

        with pytest.raises(TypeError):
            request.headers["X-Test"] = "b"
        with pytest.raises(TypeError):
            del request.headers["X-Test"]

    def test_patch_set_overrides_without_mutating_original(self) -> None:
        headers = Headers.from_items([("X-Test", "a")])
        patched = headers.patch_set({"X-Test": "b", "X-Other": "c"})

        assert headers["X-Test"] == "a"
        assert patched["X-Test"] == "b"
        assert patched["X-Other"] == "c"

    def test_partial_headers_are_immutable(self) -> None:
        headers = Headers.from_items([("Host", "example.com")])

        partial = HTTPRequestHeaders(
            method="GET",
            path="/",
            http_version="1.1",
            headers=headers,
            wire_raw_bytes=(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n"),
        )

        assert partial.headers["Host"] == "example.com"
        with pytest.raises(TypeError):
            partial.headers["Host"] = "other.example.com"


def _create_mock_reader(data_chunks: list[bytes]) -> asyncio.StreamReader:
    reader = AsyncMock(spec=asyncio.StreamReader)
    reader.read = AsyncMock(side_effect=data_chunks + [b""])
    return reader
