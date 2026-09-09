from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator

import pytest
from pytest_codspeed import BenchmarkFixture

from localstub import AsyncHTTPTestServer

RESPONSE_OBJECT = {"ok": True, "items": [1, 2, 3]}
RESPONSE_BODY = json.dumps(RESPONSE_OBJECT).encode("utf-8")
REQUEST = (
    b"GET /v1/items?page=2 HTTP/1.1\r\n"
    b"Host: localhost\r\n"
    b"User-Agent: localstub-bench/1.0\r\n"
    b"Accept: application/json\r\n"
    b"\r\n"
)

type Connection = tuple[asyncio.StreamReader, asyncio.StreamWriter]


@pytest.fixture
def connection(loop: asyncio.AbstractEventLoop) -> Iterator[Connection]:
    # One keep-alive connection is opened outside the measured region so
    # each sample is a single request/response exchange through the whole
    # server stack, without connect or accept costs.
    server = AsyncHTTPTestServer()
    server.set_json_response(RESPONSE_OBJECT)
    loop.run_until_complete(server.start())
    reader, writer = loop.run_until_complete(
        asyncio.open_connection(server.host, server.port)
    )
    try:
        yield reader, writer
    finally:
        writer.close()
        loop.run_until_complete(writer.wait_closed())
        loop.run_until_complete(server.aclose())


async def exchange(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> tuple[bytes, bytes]:
    # A raw client keeps HTTP client libraries out of the measurement.
    # The response is static, so its body length is known up front.
    writer.write(REQUEST)
    await writer.drain()
    head = await reader.readuntil(b"\r\n\r\n")
    body = await reader.readexactly(len(RESPONSE_BODY))
    return head, body


def test_server_roundtrip_static_json(
    benchmark: BenchmarkFixture,
    loop: asyncio.AbstractEventLoop,
    connection: Connection,
) -> None:
    reader, writer = connection

    def exchange_once() -> tuple[bytes, bytes]:
        return loop.run_until_complete(exchange(reader, writer))

    head, body = benchmark(exchange_once)

    assert head.startswith(b"HTTP/1.1 200 OK\r\n")
    assert body == RESPONSE_BODY
