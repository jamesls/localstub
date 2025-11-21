import asyncio
import time
from unittest.mock import AsyncMock

import pytest

from localstub.server import (
    AsyncHTTPTestServer,
    HTTPRequest,
    HTTPResponse,
    ImmediateTransmission,
    ThrottledTransmission,
)


def test_http_request_json_body_with_none_body():
    request = HTTPRequest(body=None)
    assert request.json_body is None


def test_http_request_json_body_with_empty_string():
    request = HTTPRequest(body="")
    assert request.json_body is None


def test_http_request_json_body_with_valid_json():
    request = HTTPRequest(body='{"key": "value"}')
    assert request.json_body == {"key": "value"}


def test_http_response_json_with_custom_headers():
    response = HTTPResponse.json(
        {"data": "test"},
        status=201,
        headers={"X-Custom": "value"},
    )
    assert response.status == 201
    assert response.headers["X-Custom"] == "value"
    assert response.headers["Content-Type"] == "application/json"
    assert "Content-Length" in response.headers


def test_http_response_text_with_custom_headers():
    response = HTTPResponse.text(
        "test text",
        status=202,
        headers={"X-Custom": "header"},
    )
    assert response.status == 202
    assert response.headers["X-Custom"] == "header"
    assert response.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert "Content-Length" in response.headers


def test_http_response_raw_with_custom_headers():
    response = HTTPResponse.raw(
        b"raw data",
        status=203,
        headers={"X-Custom": "raw"},
    )
    assert response.status == 203
    assert response.headers["X-Custom"] == "raw"
    assert "Content-Length" in response.headers


def test_server_url_raises_when_not_started():
    server = AsyncHTTPTestServer()
    with pytest.raises(RuntimeError, match="Server not started yet"):
        _ = server.url


def test_server_handler_getter():
    def handler(req):
        return HTTPResponse.json({"test": "value"})

    server = AsyncHTTPTestServer(handler=handler)
    assert server.handler is handler


@pytest.mark.asyncio
async def test_immediate_transmission_sends_all_at_once():
    """ImmediateTransmission should send entire body in one write."""
    strategy = ImmediateTransmission()
    body = b"x" * 10000

    writer = AsyncMock(spec=asyncio.StreamWriter)
    conn_sent = bytearray()

    await strategy.write_body(writer, body, conn_sent)

    # Should have exactly one write call with the full body
    assert writer.write.call_count == 1
    assert writer.write.call_args[0][0] == body
    assert writer.drain.call_count == 1
    assert bytes(conn_sent) == body


@pytest.mark.asyncio
async def test_immediate_transmission_without_tracking():
    """ImmediateTransmission should work without connection tracking."""
    strategy = ImmediateTransmission()
    body = b"test data"

    writer = AsyncMock(spec=asyncio.StreamWriter)

    await strategy.write_body(writer, body, None)

    assert writer.write.call_count == 1
    assert writer.write.call_args[0][0] == body
    assert writer.drain.call_count == 1


@pytest.mark.asyncio
async def test_throttled_transmission_chunks_body():
    """ThrottledTransmission should send body in multiple chunks."""
    strategy = ThrottledTransmission(chunk_size=100, delay=0.01)
    body = b"x" * 250  # Should create 3 chunks: 100, 100, 50

    writer = AsyncMock(spec=asyncio.StreamWriter)
    conn_sent = bytearray()

    start = time.time()
    await strategy.write_body(writer, body, conn_sent)
    elapsed = time.time() - start

    # Should have 3 write calls
    assert writer.write.call_count == 3
    assert writer.drain.call_count == 3

    # Verify chunk sizes
    chunks = [call[0][0] for call in writer.write.call_args_list]
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert len(chunks[2]) == 50

    # Verify connection tracking accumulated all chunks
    assert bytes(conn_sent) == body

    # Should have delayed twice (not after last chunk)
    # 2 delays * 0.01s = ~0.02s (allow some margin)
    assert elapsed >= 0.018


@pytest.mark.asyncio
async def test_throttled_transmission_single_chunk():
    """ThrottledTransmission with body smaller than chunk_size."""
    strategy = ThrottledTransmission(chunk_size=1000, delay=0.01)
    body = b"small"

    writer = AsyncMock(spec=asyncio.StreamWriter)
    conn_sent = bytearray()

    start = time.time()
    await strategy.write_body(writer, body, conn_sent)
    elapsed = time.time() - start

    # Should have exactly one write, no delays
    assert writer.write.call_count == 1
    assert writer.drain.call_count == 1
    assert bytes(conn_sent) == body

    # Should NOT have delayed (single chunk)
    assert elapsed < 0.005


@pytest.mark.asyncio
async def test_throttled_transmission_exact_multiple():
    """ThrottledTransmission when body is exact multiple of chunk_size."""
    strategy = ThrottledTransmission(chunk_size=100, delay=0.01)
    body = b"x" * 200  # Exactly 2 chunks

    writer = AsyncMock(spec=asyncio.StreamWriter)
    conn_sent = bytearray()

    start = time.time()
    await strategy.write_body(writer, body, conn_sent)
    elapsed = time.time() - start

    # Should have 2 write calls
    assert writer.write.call_count == 2
    assert writer.drain.call_count == 2

    # Verify chunks are equal size
    chunks = [call[0][0] for call in writer.write.call_args_list]
    assert len(chunks[0]) == 100
    assert len(chunks[1]) == 100
    assert bytes(conn_sent) == body

    # Should have delayed once (between chunks, not after last)
    assert elapsed >= 0.008


def test_throttled_transmission_rejects_zero_chunk_size():
    """Guard against zero chunk sizes to prevent infinite loop."""
    with pytest.raises(
        ValueError, match="chunk_size must be a positive integer"
    ):
        ThrottledTransmission(chunk_size=0, delay=0.01)


def test_throttled_transmission_rejects_negative_chunk_size():
    """Guard against negative chunk sizes to prevent hang."""
    with pytest.raises(
        ValueError, match="chunk_size must be a positive integer"
    ):
        ThrottledTransmission(chunk_size=-5, delay=0.01)
