from __future__ import annotations

import asyncio
import ssl
import time

import httpx
import pytest

from localstub import (
    AsyncHTTPTestServer,
    AsyncTLSInterceptProxy,
    CloseConnection,
    DropConnection,
    FaultyTransmission,
    HTTPResponse,
    ResponderContext,
    ResponseSpec,
    SenderContext,
    SenderNext,
    SendResult,
)
from localstub.recording import TrafficRecorder

TIMEOUT = 2.0
URL = "https://example.com/"
RESPONSE_BODY = b"0123456789" * 4


def _proxy_client(proxy: AsyncTLSInterceptProxy) -> httpx.AsyncClient:
    host, port = proxy.address
    verify = ssl.create_default_context(cafile=str(proxy.ca.ca_pem_path()))
    return httpx.AsyncClient(
        proxy=f"http://{host}:{port}", verify=verify, http2=False
    )


async def _timed_aclose(proxy: AsyncTLSInterceptProxy) -> float:
    started = time.monotonic()
    await asyncio.wait_for(proxy.aclose(), timeout=5.0)
    return time.monotonic() - started


@pytest.mark.parametrize("explicit_recorder", [False, True])
@pytest.mark.asyncio
async def test_close_connection_fault_observed_through_intercept_proxy(
    explicit_recorder: bool,
) -> None:
    recorder = TrafficRecorder() if explicit_recorder else None
    async with (
        AsyncHTTPTestServer(recorder=recorder) as server,
        AsyncTLSInterceptProxy(server=server, recorder=recorder) as proxy,
        _proxy_client(proxy) as client,
    ):
        server.set_response_sequence([
            CloseConnection(),
            HTTPResponse.json({"ok": True}),
        ])

        with pytest.raises(httpx.HTTPError):
            await client.get(URL)
        closed = await proxy.next_closed_connection(timeout=TIMEOUT)
        assert proxy.last_closed_connection is server.last_closed_connection
        assert proxy.last_closed_connection is closed
        assert proxy.closed_connections is server.closed_connections
        assert await proxy.next_request(timeout=TIMEOUT) is server.requests[0]
        assert (
            await proxy.next_exchange(timeout=TIMEOUT) is server.exchanges[0]
        )
        response = await client.get(URL)

    assert response.json() == {"ok": True}
    assert closed.reason == "close_response"
    assert closed.phase == "response"
    assert proxy.closed_connections[0] is closed
    assert proxy.dropped_closed_connections == 0
    exchange = server.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is None
    assert closed.bytes_read == len(exchange.request.wire_raw_bytes)
    assert closed.bytes_consumed == closed.bytes_read
    assert closed.bytes_written == 0


@pytest.mark.asyncio
async def test_drop_connection_fault_observed_through_intercept_proxy() -> (
    None
):
    recorder = TrafficRecorder()
    async with (
        AsyncHTTPTestServer(recorder=recorder) as server,
        AsyncTLSInterceptProxy(server=server, recorder=recorder) as proxy,
        _proxy_client(proxy) as client,
    ):
        server.set_raw_response(RESPONSE_BODY)
        server.set_transmission_strategy(
            FaultyTransmission([DropConnection(after_bytes=20)])
        )

        with pytest.raises(httpx.HTTPError):
            await client.get(URL)
        closed = await proxy.next_closed_connection(timeout=TIMEOUT)

    assert closed.reason == "response_aborted"
    assert closed.phase == "response_body"
    assert proxy.last_closed_connection is closed
    exchange = server.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is not None
    assert exchange.response.wire_raw_bytes.endswith(RESPONSE_BODY[:20])
    assert closed.bytes_written == len(exchange.response.wire_raw_bytes)
    assert closed.bytes_read == len(exchange.request.wire_raw_bytes)
    assert closed.bytes_consumed == closed.bytes_read


@pytest.mark.parametrize("start_server", [True, False])
@pytest.mark.asyncio
async def test_proxy_aclose_records_shutdown_for_idle_connection(
    start_server: bool,
) -> None:
    recorder = TrafficRecorder()
    server = AsyncHTTPTestServer(recorder=recorder)
    server.set_json_response({"ok": True})
    if start_server:
        await server.start()
    proxy = AsyncTLSInterceptProxy(server=server, recorder=recorder)
    await proxy.start()
    try:
        async with _proxy_client(proxy) as client:
            response = await client.get(URL)
            elapsed = await _timed_aclose(proxy)
        closed = await proxy.next_closed_connection(timeout=TIMEOUT)
    finally:
        await proxy.aclose()
        await server.aclose()

    assert response.status_code == 200
    # An idle client is not reading, so it never answers the TLS close
    # handshake; shutdown must not wait for it (or for the proxy's grace
    # period) before the connection loop finishes.
    assert elapsed < 0.5, f"proxy.aclose() took {elapsed:.2f}s"
    assert closed.reason == "shutdown"
    assert closed.phase == "idle"
    assert closed.requests_completed == 1
    assert server.exchanges[0].closed is None


@pytest.mark.asyncio
async def test_idle_timeout_event_published_while_tls_client_is_pooled() -> (
    None
):
    recorder = TrafficRecorder()
    async with (
        AsyncHTTPTestServer(recorder=recorder) as server,
        AsyncTLSInterceptProxy(server=server, recorder=recorder) as proxy,
        _proxy_client(proxy) as client,
    ):
        server.set_json_response({"ok": True})
        server.set_keep_alive(timeout=0.05)

        response = await client.get(URL)
        # A pooled client is not reading, so it never answers the TLS
        # close handshake; the event must not wait for it.
        closed = await proxy.next_closed_connection(timeout=TIMEOUT)

    assert response.status_code == 200
    assert closed.reason == "idle_timeout"
    assert closed.phase == "idle"
    assert closed.requests_completed == 1
    assert proxy.closed_connections == [closed]
    assert server.exchanges[0].closed is None


@pytest.mark.asyncio
async def test_proxy_aclose_after_idle_timeout_ignores_tls_peer() -> None:
    recorder = TrafficRecorder()
    async with AsyncHTTPTestServer(recorder=recorder) as server:
        server.set_json_response({"ok": True})
        server.set_keep_alive(timeout=0.05)
        proxy = AsyncTLSInterceptProxy(server=server, recorder=recorder)
        await proxy.start()
        try:
            async with _proxy_client(proxy) as client:
                response = await client.get(URL)
                closed = await proxy.next_closed_connection(timeout=TIMEOUT)
                elapsed = await _timed_aclose(proxy)
        finally:
            await proxy.aclose()

    assert response.status_code == 200
    assert elapsed < 0.5, f"proxy.aclose() took {elapsed:.2f}s"
    assert closed.reason == "idle_timeout"
    assert proxy.closed_connections == [closed]


@pytest.mark.parametrize("start_server", [True, False])
@pytest.mark.asyncio
async def test_proxy_aclose_interrupts_held_close_delay(
    start_server: bool,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def held_sleep(_: float) -> None:
        entered.set()
        await release.wait()

    recorder = TrafficRecorder()
    server = AsyncHTTPTestServer(
        recorder=recorder,
        sleep=held_sleep,
        default_response=CloseConnection(delay=5.0),
    )
    if start_server:
        await server.start()
    proxy = AsyncTLSInterceptProxy(server=server, recorder=recorder)
    await proxy.start()
    try:
        async with _proxy_client(proxy) as client:
            request = asyncio.create_task(client.get(URL))
            await asyncio.wait_for(entered.wait(), timeout=TIMEOUT)
            elapsed = await _timed_aclose(proxy)
            release.set()
            with pytest.raises(httpx.HTTPError):
                await asyncio.wait_for(request, timeout=TIMEOUT)
        closed = await proxy.next_closed_connection(timeout=TIMEOUT)
    finally:
        release.set()
        await proxy.aclose()
        await server.aclose()

    assert elapsed < 0.5
    assert closed.reason == "shutdown"
    assert closed.phase == "response"
    assert closed.requests_completed == 1
    exchange = server.exchanges[0]
    assert exchange.closed is closed
    assert exchange.response is None


@pytest.mark.asyncio
async def test_earlier_fault_decision_wins_over_proxy_shutdown() -> None:
    held = asyncio.Event()
    release = asyncio.Event()

    async def hold_after_close(
        ctx: SenderContext,
        response: ResponseSpec,
        call_next: SenderNext,
    ) -> SendResult:
        _ = ctx
        result = await call_next(response)
        if result.closed is not None:
            held.set()
            await release.wait()
        return result

    recorder = TrafficRecorder()
    async with AsyncHTTPTestServer(
        recorder=recorder, default_response=CloseConnection()
    ) as server:
        server.use_sender(hold_after_close)
        proxy = AsyncTLSInterceptProxy(server=server, recorder=recorder)
        await proxy.start()
        try:
            async with _proxy_client(proxy) as client:
                request = asyncio.create_task(client.get(URL))
                await asyncio.wait_for(held.wait(), timeout=TIMEOUT)
                elapsed = await _timed_aclose(proxy)
                release.set()
                with pytest.raises(httpx.HTTPError):
                    await asyncio.wait_for(request, timeout=TIMEOUT)
            closed = await proxy.next_closed_connection(timeout=TIMEOUT)
        finally:
            release.set()
            await proxy.aclose()

    assert elapsed < 0.5
    assert closed.reason == "close_response"
    assert closed.phase == "response"
    assert server.exchanges[0].closed is closed


@pytest.mark.asyncio
async def test_closing_one_proxy_leaves_other_proxy_connections_open() -> None:
    recorder = TrafficRecorder()
    async with AsyncHTTPTestServer(recorder=recorder) as server:
        server.set_json_response({"ok": True})
        first_proxy = AsyncTLSInterceptProxy(server=server, recorder=recorder)
        await first_proxy.start()
        try:
            async with (
                AsyncTLSInterceptProxy(
                    server=server, recorder=recorder
                ) as second_proxy,
                _proxy_client(first_proxy) as first_client,
                _proxy_client(second_proxy) as second_client,
            ):
                await first_client.get(URL)
                await second_client.get(URL)
                await _timed_aclose(first_proxy)
                closed = await first_proxy.next_closed_connection(
                    timeout=TIMEOUT
                )
                events_after_first_close = len(recorder.closed_connections)
                response = await second_client.get(URL)
        finally:
            await first_proxy.aclose()

    assert response.status_code == 200
    assert closed.reason == "shutdown"
    assert events_after_first_close == 1
    clients = [request.client for request in server.requests]
    assert closed.client == clients[0]
    assert clients[1] == clients[2]
    assert clients[0] != clients[1]


@pytest.mark.asyncio
async def test_handler_closing_proxy_on_own_connection_is_not_cancelled() -> (
    None
):
    handler_finished = asyncio.Event()
    recorder = TrafficRecorder()
    async with AsyncHTTPTestServer(recorder=recorder) as server:
        proxy = AsyncTLSInterceptProxy(server=server, recorder=recorder)
        await proxy.start()

        async def shutdown_handler(_: ResponderContext) -> HTTPResponse:
            await proxy.aclose()
            handler_finished.set()
            return HTTPResponse.text("closing")

        server.handler = shutdown_handler
        try:
            async with _proxy_client(proxy) as client:
                with pytest.raises(httpx.HTTPError):
                    await asyncio.wait_for(client.get(URL), timeout=5.0)
            await asyncio.wait_for(handler_finished.wait(), timeout=TIMEOUT)
            closed = await proxy.next_closed_connection(timeout=TIMEOUT)
        finally:
            await proxy.aclose()

    assert closed.reason == "shutdown"
    assert closed.phase == "response"
    assert closed.requests_completed == 1
    assert server.exchanges[0].closed is closed
