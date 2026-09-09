from __future__ import annotations

import asyncio

from pytest_codspeed import BenchmarkFixture

from localstub.http.request import AsyncRequestParser, ParsedRequest

SIMPLE_GET = (
    b"GET /v1/items?page=2 HTTP/1.1\r\n"
    b"Host: localhost:8080\r\n"
    b"User-Agent: localstub-bench/1.0\r\n"
    b"Accept: application/json\r\n"
    b"Accept-Encoding: gzip, deflate\r\n"
    b"Accept-Language: en-US\r\n"
    b"Authorization: Bearer 0123456789abcdef\r\n"
    b"Connection: keep-alive\r\n"
    b"X-Amz-Date: 20260909T120000Z\r\n"
    b"X-Amz-Target: Service.Operation\r\n"
    b"X-Request-Id: 0123456789abcdef\r\n"
    b"\r\n"
)
SIMPLE_GET_HEADER_COUNT = 10

CHUNK_SIZE = 1024
CHUNK_COUNT = 64
CHUNK = b"x" * CHUNK_SIZE
CHUNKED_BODY = CHUNK * CHUNK_COUNT
CHUNKED_POST = (
    b"POST /v1/upload HTTP/1.1\r\n"
    b"Host: localhost:8080\r\n"
    b"User-Agent: localstub-bench/1.0\r\n"
    b"Content-Type: application/octet-stream\r\n"
    b"Transfer-Encoding: chunked\r\n"
    b"Trailer: X-Checksum\r\n"
    b"\r\n"
    + b"".join(
        f"{CHUNK_SIZE:x}\r\n".encode("ascii") + CHUNK + b"\r\n"
        for _ in range(CHUNK_COUNT)
    )
    + b"0\r\n"
    b"X-Checksum: 0123456789abcdef\r\n"
    b"\r\n"
)


async def parse_request(wire: bytes) -> tuple[ParsedRequest | None, bytes]:
    # parse() consumes both the reader and the parser, so they are
    # rebuilt for every sample; feeding prebuilt bytes into a reader is
    # a small constant next to the parse itself.
    reader = asyncio.StreamReader()
    reader.feed_data(wire)
    reader.feed_eof()
    return await AsyncRequestParser().parse(reader)


def test_parse_simple_get(
    benchmark: BenchmarkFixture,
    loop: asyncio.AbstractEventLoop,
) -> None:
    def parse_once() -> tuple[ParsedRequest | None, bytes]:
        return loop.run_until_complete(parse_request(SIMPLE_GET))

    parsed, wire = benchmark(parse_once)

    assert parsed is not None
    assert parsed.is_complete
    assert parsed.method == "GET"
    assert parsed.url == b"/v1/items?page=2"
    assert len(parsed.headers) == SIMPLE_GET_HEADER_COUNT
    assert parsed.body == b""
    assert wire == SIMPLE_GET


def test_parse_chunked_post_with_trailer(
    benchmark: BenchmarkFixture,
    loop: asyncio.AbstractEventLoop,
) -> None:
    def parse_once() -> tuple[ParsedRequest | None, bytes]:
        return loop.run_until_complete(parse_request(CHUNKED_POST))

    parsed, wire = benchmark(parse_once)

    assert parsed is not None
    assert parsed.is_complete
    assert parsed.method == "POST"
    assert parsed.body == CHUNKED_BODY
    assert wire == CHUNKED_POST
