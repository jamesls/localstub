from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import pytest

from localstub import (
    AbortTransmission,
    AsyncHTTPTestServer,
    HeaderContext,
    HeaderDecision,
    HeaderNext,
    HTTPResponse,
    Pause,
)
from localstub.http.responsespec import StreamItem
from localstub.http.tcp import reset_stream

TIMEOUT = 2.0
# Bounds endless test streams so a server that buffers fails, not OOMs.
MAX_STREAM_EVENTS = 500
GET = b"GET /v1/messages HTTP/1.1\r\nHost: localhost\r\n\r\n"
EVENT_1 = b'event: message_start\ndata: {"n": 1}\n\n'
EVENT_2 = b'event: delta\ndata: {"n": 2}\n\n'
EVENT_3 = b"data: line one\ndata: line two\n\n"


async def _open(
    server: AsyncHTTPTestServer, request: bytes = GET
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, bytes]:
    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(request)
    await writer.drain()
    head = await asyncio.wait_for(
        reader.readuntil(b"\r\n\r\n"), timeout=TIMEOUT
    )
    return reader, writer, head


def _chunk(data: bytes) -> bytes:
    return b"%x\r\n%s\r\n" % (len(data), data)


@pytest.mark.asyncio
async def test_sse_event_reaches_client_before_gated_event_is_sent() -> None:
    release = asyncio.Event()
    async with AsyncHTTPTestServer() as server:
        server.set_default_response(
            HTTPResponse.sse([
                ("message_start", {"n": 1}),
                release,
                ("delta", {"n": 2}),
                (None, "line one\nline two"),
            ])
        )
        reader, writer, head = await _open(server)
        first = await asyncio.wait_for(
            reader.readexactly(len(_chunk(EVENT_1))), timeout=TIMEOUT
        )
        # Event 2 is held back until the gate opens.
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(reader.read(1), timeout=0.2)
        release.set()
        rest = _chunk(EVENT_2) + _chunk(EVENT_3) + b"0\r\n\r\n"
        tail = await asyncio.wait_for(
            reader.readexactly(len(rest)), timeout=TIMEOUT
        )
        exchange = await server.next_exchange(timeout=TIMEOUT)
        writer.close()

    assert first == _chunk(EVENT_1)
    assert tail == rest
    assert b"Content-Type: text/event-stream\r\n" in head
    assert b"Transfer-Encoding: chunked\r\n" in head
    assert b"Content-Length" not in head
    assert exchange.response is not None
    assert exchange.response.body == EVENT_1 + EVENT_2 + EVENT_3
    assert exchange.response.wire_raw_bytes == head + first + tail
    assert exchange.closed is None


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
async def test_stream_aborted_after_third_event_closes_connection(
    reset: bool,
) -> None:
    async def events() -> AsyncIterator[StreamItem]:
        for event in (EVENT_1, EVENT_2, EVENT_3):
            yield event
            yield Pause(0.01)
        yield AbortTransmission(reset=reset)
        yield b"never sent"

    async with AsyncHTTPTestServer() as server:
        server.handler = lambda _: HTTPResponse.stream(
            events(), headers={"Content-Type": "text/event-stream"}
        )
        reader, writer, head = await _open(server)
        body = b""
        observed_reset = False
        try:
            while chunk := await asyncio.wait_for(
                reader.read(65536), timeout=TIMEOUT
            ):
                body += chunk
        except ConnectionResetError:
            observed_reset = True
        closed = await server.next_closed_connection(timeout=TIMEOUT)
        exchange = await server.next_exchange(timeout=TIMEOUT)
        writer.close()

    sent = _chunk(EVENT_1) + _chunk(EVENT_2) + _chunk(EVENT_3)
    # A reset may discard bytes the client had not read yet.
    assert body == sent or (reset and sent.startswith(body))
    assert b"0\r\n\r\n" not in body
    assert observed_reset == reset
    assert closed.reason == "response_aborted"
    assert closed.phase == "response_body"
    assert closed.reset == reset
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.body == EVENT_1 + EVENT_2 + EVENT_3
    assert exchange.response.wire_raw_bytes == head + sent


@pytest.mark.asyncio
async def test_close_delimited_stream_ends_with_connection_close() -> None:
    async def items() -> AsyncIterator[StreamItem]:
        for item in (b"one,", "", "two"):
            yield item

    async with AsyncHTTPTestServer() as server:
        server.handler = lambda _: HTTPResponse.stream(items(), chunked=False)
        reader, writer, head = await _open(server)
        body = await asyncio.wait_for(reader.read(), timeout=TIMEOUT)
        exchange = await server.next_exchange(timeout=TIMEOUT)
        writer.close()

    assert body == b"one,two"
    assert b"Connection: close\r\n" in head
    assert b"Content-Length" not in head
    assert b"Transfer-Encoding" not in head
    assert exchange.response is not None
    assert exchange.response.body == b"one,two"


@pytest.mark.asyncio
async def test_stream_with_content_length_is_sent_unframed() -> None:
    response = HTTPResponse.stream(
        [b"ab", b"cd"], headers={"Content-Length": "4"}
    )
    async with AsyncHTTPTestServer() as server, httpx.AsyncClient() as client:
        server.set_default_response(response)
        first = await client.get(server.url)
        # A list body can be served again.
        second = await client.get(server.url)
        head = await client.head(server.url)

    assert first.content == second.content == b"abcd"
    assert "transfer-encoding" not in first.headers
    assert head.content == b""
    assert server.exchanges[-1].response is not None
    assert server.exchanges[-1].response.body == b""


@pytest.mark.asyncio
async def test_client_disconnect_mid_stream_records_partial_body() -> None:
    sent_first = asyncio.Event()
    client_gone = asyncio.Event()

    async def events() -> AsyncIterator[StreamItem]:
        yield EVENT_1
        sent_first.set()
        yield client_gone
        # Give the server's transport time to see the reset.
        yield Pause(0.1)
        for _ in range(MAX_STREAM_EVENTS):
            yield EVENT_2
            yield Pause(0.01)

    async with AsyncHTTPTestServer() as server:
        server.handler = lambda _: HTTPResponse.stream(events())
        _, writer, _ = await _open(server)
        await asyncio.wait_for(sent_first.wait(), timeout=TIMEOUT)
        # A reset, not a FIN, so the server learns the client is gone
        # before it writes again.
        reset_stream(writer)
        client_gone.set()
        closed = await server.next_closed_connection(timeout=TIMEOUT)
        exchange = await server.next_exchange(timeout=TIMEOUT)

    assert closed.reason == "client"
    assert closed.phase == "response_body"
    assert exchange.response is not None
    # Nothing written after the client left counts as sent.
    assert exchange.response.body == EVENT_1


@pytest.mark.asyncio
async def test_header_middleware_rejects_streamed_body() -> None:
    async def middleware(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        with pytest.raises(ValueError, match="streamed body"):
            await ctx.send(HTTPResponse.stream([b"x"]))
        return await call_next()

    async with AsyncHTTPTestServer() as server, httpx.AsyncClient() as client:
        server.use_headers(middleware)
        response = await client.get(server.url)

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_bodiless_streamed_responses_frame_no_body() -> None:
    async with AsyncHTTPTestServer() as server:
        server.set_default_response(HTTPResponse.stream([b"x"], status=204))
        reader, writer, no_content = await _open(server)
        # The connection stays open for the HEAD request.
        server.set_default_response(HTTPResponse.stream([b"x"], chunked=False))
        writer.write(b"HEAD / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        head = await asyncio.wait_for(
            reader.readuntil(b"\r\n\r\n"), timeout=TIMEOUT
        )
        writer.close()

    assert no_content.startswith(b"HTTP/1.1 204 ")
    assert b"Transfer-Encoding" not in no_content
    assert b"Connection: close" not in head
    assert [e.response.body for e in server.exchanges if e.response] == [
        b"",
        b"",
    ]
