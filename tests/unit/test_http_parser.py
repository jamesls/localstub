"""Unit tests for the http_parser module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from localstub.http_parser import (
    AsyncRequestParser,
    AsyncResponseParser,
    ParsedRequest,
    ParsedResponse,
    RequestProtocol,
    ResponseProtocol,
    headers_to_message,
)


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
        assert req.is_complete is False

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

        When two HTTP requests arrive in the same TCP frame, httptools
        parses them both synchronously. The parser must preserve the first
        completed request rather than overwriting it with the second.
        """
        # Two complete requests in one buffer
        pipelined_data = (
            b"GET /first HTTP/1.1\r\n"
            b"Host: first.com\r\n"
            b"\r\n"
            b"GET /second HTTP/1.1\r\n"
            b"Host: second.com\r\n"
            b"\r\n"
        )
        reader = _create_mock_reader([pipelined_data])

        parser = AsyncRequestParser()
        parsed, wire_bytes = await parser.parse(reader)

        # First request should be returned, not the second
        assert parsed is not None
        assert parsed.method == "GET"
        assert parsed.url == b"/first"
        assert parsed.is_complete is True
        # Headers should be from first request only
        header_names = [h[0] for h in parsed.headers]
        assert b"Host" in header_names
        host_value = next(h[1] for h in parsed.headers if h[0] == b"Host")
        assert host_value == b"first.com"


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


def _create_mock_reader(data_chunks: list[bytes]) -> asyncio.StreamReader:
    """Create a mock StreamReader that returns data chunks then EOF."""
    reader = AsyncMock(spec=asyncio.StreamReader)
    # Return each chunk, then empty bytes for EOF
    reader.read = AsyncMock(side_effect=data_chunks + [b""])
    return reader
