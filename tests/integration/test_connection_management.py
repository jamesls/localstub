import asyncio

import pytest

from localstub.forward import Forwarder
from localstub.middleware import ResponderContext, ResponderNext, ResponseSpec
from localstub.server import (
    AsyncHTTPTestServer,
    HTTPResponse,
    ThrottledTransmission,
)


async def _read_http_response(reader: asyncio.StreamReader) -> bytes:
    header_bytes = await asyncio.wait_for(
        reader.readuntil(b"\r\n\r\n"),
        timeout=1.0,
    )
    content_length = 0
    for line in header_bytes.split(b"\r\n"):
        if line.lower().startswith(b"content-length:"):
            length_value = line.split(b":", 1)[1].strip()
            content_length = int(length_value) if length_value else 0
            break

    if content_length == 0:
        return header_bytes

    body_bytes = await asyncio.wait_for(
        reader.readexactly(content_length),
        timeout=1.0,
    )
    return header_bytes + body_bytes


async def _assert_connection_closes(reader: asyncio.StreamReader) -> None:
    try:
        eof = await asyncio.wait_for(reader.read(1), timeout=0.5)
    except TimeoutError as exc:
        raise AssertionError(
            "Expected server to close the TCP connection (EOF)."
        ) from exc
    assert eof == b""


@pytest.mark.asyncio
async def test_aclose_closes_just_accepted_client_connection() -> None:
    server = AsyncHTTPTestServer()
    await server.start()
    reader, writer = await asyncio.open_connection(server.host, server.port)

    try:
        async with asyncio.timeout(0.5):
            await server.aclose()
        eof = await asyncio.wait_for(reader.read(1), timeout=0.5)
        assert eof == b""
    finally:
        writer.close()
        await asyncio.wait_for(writer.wait_closed(), timeout=0.5)
        await server.aclose()


@pytest.mark.asyncio
async def test_aclose_stops_active_handler_before_returning() -> None:
    handler_started = asyncio.Event()
    handler_release = asyncio.Event()
    handler_finished = asyncio.Event()
    state_mutated = False

    async def handler(_: ResponderContext) -> HTTPResponse:
        nonlocal state_mutated
        handler_started.set()
        try:
            await handler_release.wait()
            state_mutated = True
            return HTTPResponse.text("late response")
        finally:
            handler_finished.set()

    server = AsyncHTTPTestServer(handler=handler)
    await server.start()
    _, writer = await asyncio.open_connection(server.host, server.port)

    try:
        writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        await asyncio.wait_for(handler_started.wait(), timeout=0.5)

        await asyncio.wait_for(server.aclose(), timeout=0.5)
        handler_release.set()
        await asyncio.wait_for(handler_finished.wait(), timeout=0.5)

        assert not state_mutated
    finally:
        handler_release.set()
        writer.close()
        await asyncio.wait_for(writer.wait_closed(), timeout=0.5)
        await server.aclose()


@pytest.mark.asyncio
async def test_server_closes_when_response_has_connection_close():
    def handler(request):
        return HTTPResponse.text(
            "ok",
            headers={"Connection": "close"},
        )

    async with AsyncHTTPTestServer(handler=handler) as server:
        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        try:
            writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            response = await _read_http_response(reader)
            assert b"Connection: close" in response
            await _assert_connection_closes(reader)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass

        assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_server_closes_http10_connection_by_default():
    async with AsyncHTTPTestServer() as server:
        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        try:
            writer.write(b"GET / HTTP/1.0\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            await _read_http_response(reader)
            await _assert_connection_closes(reader)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass

        assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("method", "status"),
    [("HEAD", 200), ("GET", 204), ("GET", 304)],
)
@pytest.mark.parametrize("throttled", [False, True])
@pytest.mark.asyncio
async def test_bodyless_response_preserves_keep_alive_connection(
    method: str,
    status: int,
    throttled: bool,
) -> None:
    def handler(ctx: ResponderContext) -> HTTPResponse:
        if ctx.request.path == "/bodyless":
            return HTTPResponse(status=status, body=b"unexpected-body")
        return HTTPResponse.text("next-response")

    async with AsyncHTTPTestServer(handler=handler) as server:
        if throttled:
            server.set_transmission_strategy(
                ThrottledTransmission(chunk_size=1, delay=0)
            )

        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        try:
            writer.write(
                f"{method} /bodyless HTTP/1.1\r\n"
                "Host: localhost\r\n"
                "\r\n".encode()
            )
            await writer.drain()

            first_response = await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                timeout=1.0,
            )
            assert first_response.startswith(f"HTTP/1.1 {status} ".encode())
            if status == 204:
                assert b"content-length:" not in first_response.lower()
            else:
                assert b"Content-Length: 15\r\n" in first_response

            writer.write(b"GET /next HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            next_response = await _read_http_response(reader)
            assert next_response.startswith(b"HTTP/1.1 200 OK\r\n")
            assert next_response.endswith(b"next-response")
        finally:
            writer.close()
            await writer.wait_closed()


@pytest.mark.asyncio
async def test_rewritten_connection_close_reaches_sender_stage():
    async def rewrite_connection_close(
        ctx: ResponderContext,
        call_next: ResponderNext,
    ) -> ResponseSpec:
        next_ctx = ctx.clone_request(headers={"Connection": "close"})
        return await call_next(ctx=next_ctx)

    async with AsyncHTTPTestServer() as server:
        server.use(rewrite_connection_close)
        server.set_text_response("ok")

        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        try:
            writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()

            response = await _read_http_response(reader)
            assert b"Connection: close" in response
            await _assert_connection_closes(reader)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass

        assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_raw_forwarder_drops_connection_scoped_headers():
    captured_headers = b""
    request_received = asyncio.Event()

    async def upstream_handler(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        nonlocal captured_headers
        captured_headers = await reader.readuntil(b"\r\n\r\n")
        request_received.set()
        writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    upstream = await asyncio.start_server(upstream_handler, "127.0.0.1", 0)
    assert upstream.sockets is not None
    upstream_host, upstream_port = upstream.sockets[0].getsockname()[:2]

    try:
        forwarder = Forwarder(verify_upstream=False)
        async with AsyncHTTPTestServer(raw_forwarder=forwarder) as proxy:
            reader, writer = await asyncio.open_connection(
                proxy.host, proxy.port
            )
            try:
                request = (
                    f"GET http://{upstream_host}:{upstream_port}/ HTTP/1.1\r\n"
                    f"Host: {upstream_host}:{upstream_port}\r\n"
                    "Connection: close, x-remove-me\r\n"
                    "X-Remove-Me: secret\r\n"
                    "\r\n"
                ).encode()
                writer.write(request)
                await writer.drain()

                await reader.read(4096)
                await asyncio.wait_for(request_received.wait(), timeout=1.0)
            finally:
                writer.close()
                try:
                    await writer.wait_closed()
                except ConnectionResetError:
                    pass
    finally:
        upstream.close()
        await upstream.wait_closed()

    assert b"connection:" not in captured_headers.lower()
    assert b"x-remove-me:" not in captured_headers.lower()


@pytest.mark.asyncio
async def test_connection_close_is_parsed_as_token_not_substring():
    async with AsyncHTTPTestServer() as server:
        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        try:
            writer.write(
                b"GET /first HTTP/1.1\r\n"
                b"Host: localhost\r\n"
                b"Connection: disclose\r\n"
                b"\r\n"
            )
            await writer.drain()
            response = await _read_http_response(reader)
            assert b"connection: close" not in response.lower()

            writer.write(b"GET /second HTTP/1.1\r\nHost: localhost\r\n\r\n")
            await writer.drain()
            await _read_http_response(reader)
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionResetError:
                pass

        assert len(server.requests) == 2
