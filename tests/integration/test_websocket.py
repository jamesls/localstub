from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import AsyncIterator

import aiohttp
import pytest
import pytest_asyncio
import websockets

from localstub import (
    AcceptWebSocket,
    AsyncHTTPTestServer,
    AsyncTLSInterceptProxy,
    CloseConnection,
    HTTPResponse,
    Opcode,
    PerMessageDeflate,
    WebSocketClose,
    WebSocketFrame,
    WebSocketSession,
)
from localstub.websocket.handshake import accept_key

WS_KEY = "dGhlIHNhbXBsZSBub25jZQ=="


async def echo(ws: WebSocketSession) -> None:
    async for message in ws:
        await ws.send(message)


@pytest_asyncio.fixture
async def server() -> AsyncIterator[AsyncHTTPTestServer]:
    async with AsyncHTTPTestServer() as srv:
        yield srv


async def _subscribe(url: str) -> list[str]:
    """A client that resumes after the last sequence number it saw."""
    seen: list[str] = []
    last = 0
    while True:
        try:
            async with websockets.connect(url, open_timeout=1) as ws:
                if last:
                    await ws.send(json.dumps({"resume_after": last}))
                async for raw in ws:
                    text = str(raw)
                    seen.append(text)
                    last = json.loads(text)["seq"]
                return seen
        except (websockets.InvalidStatus, websockets.InvalidMessage):
            await asyncio.sleep(0)
        except websockets.ConnectionClosedError:
            await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_client_reconnects_after_rejections_and_drop(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_response_sequence([
        HTTPResponse(status=503),
        CloseConnection(reset=True),
        AcceptWebSocket(),
        AcceptWebSocket(),
    ])
    task = asyncio.create_task(_subscribe(server.ws_url))

    first = await server.next_websocket(timeout=2.0)
    await first.send_text('{"seq": 1}')
    await first.drop()
    second = await server.next_websocket(timeout=2.0)
    resume = await second.receive(timeout=2.0)
    await second.send_text('{"seq": 2}')
    await second.close()
    closed = await first.wait_closed(timeout=1.0)

    assert resume.text == '{"resume_after": 1}'
    assert closed.reason == "websocket_drop"
    assert await asyncio.wait_for(task, 2.0) == [
        '{"seq": 1}',
        '{"seq": 2}',
    ]
    assert len(server.requests) == 4
    assert [e.response.status for e in server.exchanges if e.response] == [
        503,
        101,
        101,
    ]
    assert server.websockets == [first, second]
    assert server.last_websocket is second


@pytest.mark.asyncio
async def test_records_exact_client_frames(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket())

    async with websockets.connect(server.ws_url, compression=None) as client:
        ws = await server.next_websocket(timeout=1.0)
        await client.send("hello")
        await client.close()

    closed = await ws.wait_closed(timeout=1.0)
    recorded = ws.recorded
    assert closed.reason == "websocket_close"
    assert closed.phase == "websocket"
    assert not recorded.violations
    assert recorded.client_close == WebSocketClose(code=1000, reason="")
    assert all(f.frame.mask_key is not None for f in recorded.client_frames)
    assert recorded.client_frames[0].wire_raw_bytes[:2] == b"\x81\x85"
    assert server.last_exchange is not None
    assert server.last_exchange.closed is None
    assert server.closed_connections[-1] is closed
    assert closed.requests_completed == 1
    client_bytes = server.get_connection_bytes_received(ws.request.client)
    assert client_bytes is not None
    assert client_bytes.endswith(recorded.client_frames[-1].wire_raw_bytes)


@pytest.mark.asyncio
async def test_handler_acknowledges_requests_on_routed_subprotocol(
    server: AsyncHTTPTestServer,
) -> None:
    async def serve(ws: WebSocketSession) -> None:
        async for message in ws:
            request = json.loads(message.text)
            await ws.send_text(json.dumps({"ack": request["id"]}))

    server.add_route(
        "GET",
        "/rpc",
        lambda ctx: AcceptWebSocket(handler=serve, subprotocol="rpc.v1"),
    )

    async with (
        aiohttp.ClientSession() as session,
        session.ws_connect(
            f"{server.url}rpc", protocols=("rpc.v1",)
        ) as client,
    ):
        acks = []
        for request_id in (1, 2):
            await client.send_str(json.dumps({"id": request_id}))
            acks.append(json.loads((await client.receive()).data))
        assert client.protocol == "rpc.v1"

    assert acks == [{"ack": 1}, {"ack": 2}]
    ws = server.last_websocket
    assert ws is not None
    assert ws.recorded.subprotocol == "rpc.v1"
    assert (await ws.wait_closed(timeout=1.0)).reason == "websocket_close"


@pytest.mark.asyncio
async def test_unanswered_pings_make_client_give_up(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket(auto_pong=False))

    async with websockets.connect(
        server.ws_url, ping_interval=0.05, ping_timeout=0.05
    ) as client:
        ws = await server.next_websocket(timeout=1.0)
        ping = await ws.receive_frame(timeout=2.0)
        with pytest.raises(websockets.ConnectionClosed):
            await asyncio.wait_for(client.recv(), timeout=2.0)

    closed = await ws.wait_closed(timeout=2.0)
    assert ping.frame.opcode == Opcode.PING
    assert closed.reason in {"websocket_close", "client"}
    assert not [
        f for f in ws.recorded.server_frames if f.frame.opcode == Opcode.PONG
    ]


@pytest.mark.asyncio
async def test_auto_pong_keeps_aiohttp_heartbeat_alive(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket())

    async with (
        aiohttp.ClientSession() as session,
        session.ws_connect(server.ws_url, heartbeat=0.05) as client,
    ):
        ws = await server.next_websocket(timeout=1.0)
        await ws.receive_frame(timeout=2.0)
        await asyncio.sleep(0.2)
        await ws.send_text("still here")
        assert (await client.receive()).data == "still here"

    pongs = [
        f for f in ws.recorded.server_frames if f.frame.opcode == Opcode.PONG
    ]
    assert pongs


@pytest.mark.asyncio
async def test_server_close_code_reaches_client(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket())

    async with (
        aiohttp.ClientSession() as session,
        session.ws_connect(server.ws_url) as client,
    ):
        ws = await server.next_websocket(timeout=1.0)
        closing = asyncio.create_task(ws.close(1013, "try again later"))
        message = await client.receive()
        await closing

    assert message.type == aiohttp.WSMsgType.CLOSE
    assert client.close_code == 1013
    assert ws.recorded.server_close == WebSocketClose(1013, "try again later")
    assert (await ws.wait_closed(timeout=1.0)).reason == "websocket_close"


@pytest.mark.asyncio
async def test_frame_cut_short_then_reset(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket())

    async with websockets.connect(server.ws_url) as client:
        ws = await server.next_websocket(timeout=1.0)
        frame = WebSocketFrame.text("partial update")
        await ws.send_raw(frame.encode()[:5])
        await ws.drop(reset=True)
        with pytest.raises(websockets.ConnectionClosed):
            await asyncio.wait_for(client.recv(), timeout=1.0)

    closed = await ws.wait_closed(timeout=1.0)
    assert closed.reason == "websocket_drop"
    assert closed.reset
    sent = server.get_connection_bytes_sent(ws.request.client)
    assert sent is not None
    assert sent.endswith(frame.encode()[:5])


@pytest.mark.asyncio
async def test_client_fails_connection_on_wrong_accept_key(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(
        AcceptWebSocket(headers={"Sec-WebSocket-Accept": "bogus"})
    )

    with pytest.raises(websockets.InvalidHandshake):
        await websockets.connect(server.ws_url, open_timeout=1)

    ws = await server.next_websocket(timeout=1.0)
    assert (await ws.wait_closed(timeout=2.0)).reason == "client"


@pytest.mark.asyncio
async def test_rejected_handshake_names_failed_check(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket())
    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(
        b"GET / HTTP/1.1\r\nHost: x\r\nUpgrade: websocket\r\n"
        b"Connection: Upgrade\r\nSec-WebSocket-Version: 8\r\n"
        + f"Sec-WebSocket-Key: {WS_KEY}\r\n\r\n".encode()
    )
    await writer.drain()
    head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 1.0)
    writer.close()
    await writer.wait_closed()

    assert head.startswith(b"HTTP/1.1 426")
    assert b"Sec-WebSocket-Version: 13\r\n" in head
    exchange = await server.next_exchange(timeout=1.0)
    assert exchange.response is not None
    assert b"Sec-WebSocket-Version" in exchange.response.response.body
    assert server.websockets == []


@pytest.mark.parametrize("client", ["websockets", "aiohttp"])
@pytest.mark.asyncio
async def test_permessage_deflate_records_frames_and_messages(
    server: AsyncHTTPTestServer, client: str
) -> None:
    server.set_default_response(
        AcceptWebSocket(handler=echo, compression=PerMessageDeflate())
    )
    text = "compress me " * 20

    if client == "websockets":
        async with websockets.connect(server.ws_url) as conn:
            await conn.send(text)
            assert await conn.recv() == text
    else:
        async with (
            aiohttp.ClientSession() as session,
            session.ws_connect(server.ws_url, compress=15) as conn,
        ):
            await conn.send_str(text)
            assert (await conn.receive()).data == text

    ws = await server.next_websocket(timeout=1.0)
    await ws.wait_closed(timeout=1.0)
    recorded = ws.recorded
    assert recorded.compression is not None
    assert not recorded.violations
    for message in recorded.messages[:2]:
        assert message.compressed
        assert message.text == text
        assert message.frames[0].frame.rsv1
        assert len(message.frames[0].frame.payload) < len(text)


@pytest.mark.asyncio
async def test_compression_declined_without_option(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket(handler=echo))

    async with websockets.connect(server.ws_url) as conn:
        await conn.send("plain")
        assert await conn.recv() == "plain"

    ws = await server.next_websocket(timeout=1.0)
    await ws.wait_closed(timeout=1.0)
    assert ws.recorded.compression is None
    assert ws.recorded.client_frames[0].frame.payload == b"plain"


@pytest.mark.asyncio
async def test_aclose_shuts_down_open_session(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket())
    conn = await websockets.connect(server.ws_url)
    ws = await server.next_websocket(timeout=1.0)

    await server.aclose()

    closed = await ws.wait_closed(timeout=1.0)
    assert closed.reason == "shutdown"
    assert closed.phase == "websocket"
    with pytest.raises(websockets.ConnectionClosed):
        await asyncio.wait_for(conn.recv(), timeout=1.0)


@pytest.mark.asyncio
async def test_handler_that_shuts_server_down_is_not_cancelled() -> None:
    finished = asyncio.Event()

    async def handler(ws: WebSocketSession) -> None:
        _ = ws
        await srv.aclose()
        finished.set()

    srv = AsyncHTTPTestServer()
    await srv.start()
    srv.set_default_response(AcceptWebSocket(handler=handler))
    try:
        async with websockets.connect(srv.ws_url) as conn:
            with pytest.raises(websockets.ConnectionClosed):
                await asyncio.wait_for(conn.recv(), timeout=1.0)
        await asyncio.wait_for(finished.wait(), timeout=1.0)
        closed = await srv.next_closed_connection(timeout=1.0)
        assert closed.reason == "shutdown"
        assert closed.phase == "websocket"
    finally:
        await srv.aclose()


@pytest.mark.asyncio
async def test_clear_requests_resets_histories_but_open_session_records(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket())

    async with websockets.connect(server.ws_url) as conn:
        ws = await server.next_websocket(timeout=1.0)
        server.clear_requests()
        await conn.send("after clear")
        assert (await ws.receive(timeout=1.0)).text == "after clear"

    assert server.websockets == []
    assert server.last_websocket is None
    assert server.recorder.websockets == []
    assert server.dropped_websockets == 0
    assert (await ws.wait_closed(timeout=1.0)).reason == "websocket_close"


@pytest.mark.asyncio
async def test_unread_sessions_are_evicted_and_counted() -> None:
    async with AsyncHTTPTestServer(recording_buffer_size=1) as srv:
        srv.set_default_response(AcceptWebSocket())
        for _ in range(2):
            async with websockets.connect(srv.ws_url):
                pass

        assert srv.dropped_websockets == 1
        assert srv.recorder.dropped_websockets == 1
        assert len(srv.websockets) == 1
        assert await srv.next_websocket(timeout=1.0) is srv.last_websocket


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_connection_bytes", "kept"),
    [(4, ["three"]), (0, ["one", "two", "three"])],
)
async def test_max_connection_bytes_bounds_unread_messages(
    max_connection_bytes: int, kept: list[str]
) -> None:
    async with AsyncHTTPTestServer(
        max_connection_bytes=max_connection_bytes
    ) as srv:
        srv.set_default_response(AcceptWebSocket())
        async with websockets.connect(srv.ws_url) as ws:
            session = await srv.next_websocket(timeout=1.0)
            for text in ("one", "two", "three"):
                await ws.send(text)
            await ws.ping()

            received = [
                (await session.receive(timeout=1.0)).text for _ in kept
            ]

            assert received == kept
            assert session.dropped_unread_messages == 3 - len(kept)
            assert [m.text for m in session.recorded.messages] == kept


@pytest.mark.asyncio
async def test_wss_through_tls_intercept(
    server: AsyncHTTPTestServer,
) -> None:
    server.set_default_response(AcceptWebSocket(handler=echo))

    async with AsyncTLSInterceptProxy(server=server) as proxy:
        ctx = ssl.create_default_context(cafile=str(proxy.ca.ca_pem_path()))
        async with websockets.connect(
            "wss://example.com/", proxy=proxy.endpoint_url, ssl=ctx
        ) as conn:
            await conn.send("intercepted")
            assert await conn.recv() == "intercepted"

        ws = await server.next_websocket(timeout=1.0)
        closed = await ws.wait_closed(timeout=1.0)
        assert closed.reason == "websocket_close"
        assert await proxy.next_websocket(timeout=1.0) is ws.recorded
        assert proxy.last_websocket is ws.recorded
        assert proxy.websockets == [ws.recorded]
        assert proxy.dropped_websockets == 0
        assert ws.recorded.client_close == WebSocketClose(1000)


@pytest.mark.asyncio
async def test_forward_relay_matches_upstream_record_frame_for_frame() -> None:
    async with AsyncHTTPTestServer() as upstream:
        upstream.set_default_response(
            AcceptWebSocket(handler=echo, compression=PerMessageDeflate())
        )
        async with AsyncTLSInterceptProxy(
            default_mode="forward", upstream_tls=False
        ) as proxy:
            ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )
            async with websockets.connect(
                f"wss://localhost:{upstream.port}/",
                proxy=proxy.endpoint_url,
                ssl=ctx,
            ) as conn:
                await conn.send("one")
                assert await conn.recv() == "one"
                await conn.send(bytes(300))
                assert await conn.recv() == bytes(300)

            relayed = await proxy.next_websocket(timeout=2.0)
            await relayed.wait_ended(timeout=5.0)
            stub = await upstream.next_websocket(timeout=1.0)
            await stub.wait_closed(timeout=2.0)

    def frames(record: object) -> list[tuple[str, bytes]]:
        return [(f.sender, f.wire_raw_bytes) for f in record.frames]

    assert frames(relayed) == frames(stub.recorded)
    assert len(relayed.frames) == 6
    assert relayed.compression == stub.recorded.compression
    assert not relayed.violations
    assert [m.data for m in relayed.server_messages] == [b"one", bytes(300)]
    assert all(m.compressed for m in relayed.messages)
    assert relayed.client_close == WebSocketClose(1000)
    assert relayed.server_close == WebSocketClose(1000)
    assert relayed.closed is None
    assert relayed.handshake.response is not None
    assert relayed.handshake.response.status == 101


@pytest.mark.asyncio
async def test_forward_relay_flushes_bytes_sent_with_handshakes() -> None:
    server_frame = WebSocketFrame.text("pushed").encode()
    client_frame = WebSocketFrame(
        Opcode.TEXT, b"early", mask_key=b"\x01\x02\x03\x04"
    ).encode()
    received = asyncio.get_running_loop().create_future()

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
            + f"Sec-WebSocket-Accept: {accept_key(WS_KEY)}\r\n\r\n".encode()
            + server_frame
        )
        await writer.drain()
        received.set_result(await reader.readexactly(len(client_frame)))
        writer.close()

    upstream = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = upstream.sockets[0].getsockname()[1]
    try:
        async with AsyncTLSInterceptProxy(
            default_mode="forward", upstream_tls=False
        ) as proxy:
            reader, writer = await asyncio.open_connection(*proxy.address)
            writer.write(f"CONNECT 127.0.0.1:{port} HTTP/1.1\r\n\r\n".encode())
            await writer.drain()
            await reader.readuntil(b"\r\n\r\n")
            ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )
            await writer.start_tls(ctx, server_hostname="127.0.0.1")
            writer.write(
                b"GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                + f"Sec-WebSocket-Key: {WS_KEY}\r\n".encode()
                + b"Sec-WebSocket-Version: 13\r\n\r\n"
                + client_frame
            )
            await writer.drain()
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 2.0)
            pushed = await asyncio.wait_for(
                reader.readexactly(len(server_frame)), 2.0
            )
            assert await asyncio.wait_for(received, 2.0) == client_frame
            relayed = await proxy.next_websocket(timeout=1.0)
            await relayed.wait_ended(timeout=2.0)
            writer.close()
    finally:
        upstream.close()
        await upstream.wait_closed()

    assert head.startswith(b"HTTP/1.1 101")
    assert pushed == server_frame
    assert {(f.sender, f.wire_raw_bytes) for f in relayed.frames} == {
        ("client", client_frame),
        ("server", server_frame),
    }
    assert [m.text for m in relayed.client_messages] == ["early"]
