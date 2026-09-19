from __future__ import annotations

import asyncio

from pytest_codspeed import BenchmarkFixture

from localstub.http.response import AsyncMultiResponseParser, ParsedResponse

CHUNK_SIZE = 1024
CHUNK_COUNT = 64
CHUNK = b"x" * CHUNK_SIZE
CHUNKED_BODY = CHUNK * CHUNK_COUNT
CHUNKED_RESPONSE = (
    b"HTTP/1.1 200 OK\r\n"
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


async def parse_response(
    wire: bytes,
) -> tuple[ParsedResponse | None, bytes, bytes]:
    reader = asyncio.StreamReader()
    reader.feed_data(wire)
    reader.feed_eof()
    # Rebuild the consumed reader and parser for each sample. The odd read
    # size splits payloads, a chunk-size line, and a chunk-ending CRLF.
    parser = AsyncMultiResponseParser(max_read=257)
    parsed, wire_bytes = await parser.next_response(
        reader, request_method="GET"
    )
    # Materialize the body inside the measured operation: this joins the
    # decoded chunks and is part of the cost of consuming a response.
    body = b"" if parsed is None else parsed.body
    return parsed, wire_bytes, body


def test_parse_chunked_response_with_trailer(
    benchmark: BenchmarkFixture,
    loop: asyncio.AbstractEventLoop,
) -> None:
    def parse_once() -> tuple[ParsedResponse | None, bytes, bytes]:
        return loop.run_until_complete(parse_response(CHUNKED_RESPONSE))

    parsed, wire, body = benchmark(parse_once)

    assert parsed is not None
    assert parsed.is_complete
    assert parsed.status_code == 200
    assert (b"X-Checksum", b"0123456789abcdef") in parsed.headers
    assert body == CHUNKED_BODY
    assert wire == CHUNKED_RESPONSE
