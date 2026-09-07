from __future__ import annotations

import asyncio
import gc
import socket
import weakref
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.http.clients.pool import (
    ConnectionPool,
    Origin,
    PooledConnection,
)
from localstub.http.response import AsyncMultiResponseParser
from localstub.http.uri import ParsedURI

ORIGIN_A = Origin(scheme="http", host="a.example", port=80)
ORIGIN_B = Origin(scheme="http", host="b.example", port=80)
ORIGINS = (ORIGIN_A, ORIGIN_B)


class FakeClock:
    def __init__(self) -> None:
        self.current = 0.0

    def now(self) -> float:
        return self.current


class CancellationResistantReader(asyncio.StreamReader):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.proceed = asyncio.Event()

    async def read(self, n: int = -1) -> bytes:
        self.started.set()
        try:
            await self.proceed.wait()
        except asyncio.CancelledError:
            await self.proceed.wait()
        return b"unexpected idle data"


def _make_writer() -> Mock:
    writer = Mock(spec=asyncio.StreamWriter)
    writer.is_closing.return_value = False
    writer.wait_closed = AsyncMock()
    writer.get_extra_info.return_value = None

    def _close() -> None:
        writer.is_closing.return_value = True

    writer.close.side_effect = _close
    return writer


def _make_stream_pair() -> tuple[asyncio.StreamReader, Mock]:
    return asyncio.StreamReader(), _make_writer()


class FakeOpener:
    def __init__(self) -> None:
        self.dial_count = 0
        self.readers: list[asyncio.StreamReader] = []
        self.writers: list[Mock] = []

    async def __call__(
        self, origin: Origin
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        self.dial_count += 1
        reader, writer = _make_stream_pair()
        self.readers.append(reader)
        self.writers.append(writer)
        return reader, writer


def _make_connection(reader: asyncio.StreamReader) -> PooledConnection:
    return PooledConnection(
        ORIGIN_A, reader, _make_writer(), AsyncMultiResponseParser()
    )


class SocketPairOpener:
    def __init__(self) -> None:
        self.dial_count = 0
        self.peers: list[socket.socket] = []

    async def __call__(
        self, origin: Origin
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        self.dial_count += 1
        local, peer = socket.socketpair()
        self.peers.append(peer)
        return await asyncio.open_connection(sock=local)

    def close(self) -> None:
        for peer in self.peers:
            peer.close()


@asynccontextmanager
async def _socket_connection() -> AsyncIterator[
    tuple[PooledConnection, socket.socket]
]:
    opener = SocketPairOpener()
    reader, writer = await opener(ORIGIN_A)
    connection = PooledConnection(
        ORIGIN_A, reader, writer, AsyncMultiResponseParser()
    )
    try:
        yield connection, opener.peers[0]
    finally:
        connection.close()
        await connection.wait_closed()
        opener.close()


def _send_stray_byte(peer: socket.socket) -> None:
    peer.send(b"x")


def _close_peer(peer: socket.socket) -> None:
    peer.close()


@st.composite
def _release_histories(
    draw: st.DrawFn,
) -> tuple[list[int], list[int], int]:
    origin_indexes = draw(
        st.lists(
            st.integers(min_value=0, max_value=len(ORIGINS) - 1),
            min_size=1,
            max_size=8,
        )
    )
    release_order = list(draw(st.permutations(range(len(origin_indexes)))))
    idle_cap = draw(
        st.integers(min_value=0, max_value=len(origin_indexes) + 2)
    )
    return origin_indexes, release_order, idle_cap


async def _check_release_history(
    origin_indexes: list[int],
    release_order: list[int],
    idle_cap: int,
) -> None:
    opener = FakeOpener()
    clock = FakeClock()
    pool = ConnectionPool(
        opener,
        max_connections_per_origin=len(origin_indexes),
        max_idle_connections=idle_cap,
        idle_timeout=None,
        clock=clock,
    )
    connections = [
        await pool.acquire(ORIGINS[origin_index])
        for origin_index in origin_indexes
    ]

    for release_time, connection_index in enumerate(release_order):
        clock.current = float(release_time)
        await pool.release(connections[connection_index], reusable=True)

    retained_count = min(idle_cap, len(connections))
    retained_indexes = set(release_order[-retained_count:])
    if retained_count == 0:
        retained_indexes.clear()

    reused: list[PooledConnection] = []
    for origin_index, origin in enumerate(ORIGINS):
        expected_indexes = [
            connection_index
            for connection_index in reversed(release_order)
            if connection_index in retained_indexes
            and origin_indexes[connection_index] == origin_index
        ]
        for connection_index in expected_indexes:
            connection = await pool.acquire(origin)
            assert connection is connections[connection_index]
            reused.append(connection)

    assert opener.dial_count == len(connections)
    for connection_index, writer in enumerate(opener.writers):
        if connection_index in retained_indexes:
            writer.close.assert_not_called()
        else:
            writer.close.assert_called_once()

    for connection in reused:
        await pool.release(connection, reusable=False)
    await pool.aclose()

    for writer in opener.writers:
        writer.close.assert_called_once()


@pytest.mark.asyncio
@given(history=_release_histories())
async def test_pool_retains_newest_connections_for_arbitrary_release_histories(
    history: tuple[list[int], list[int], int],
) -> None:
    await _check_release_history(*history)


async def _check_expiration_history(
    origin_indexes: list[int],
    idle_timeout: int,
    advance: int,
    trigger_origin_index: int,
) -> None:
    opener = FakeOpener()
    clock = FakeClock()
    pool = ConnectionPool(
        opener,
        max_connections_per_origin=len(origin_indexes),
        max_idle_connections=len(origin_indexes),
        idle_timeout=float(idle_timeout),
        clock=clock,
    )
    connections = [
        await pool.acquire(ORIGINS[origin_index])
        for origin_index in origin_indexes
    ]
    for release_time, connection in enumerate(connections):
        clock.current = float(release_time)
        await pool.release(connection, reusable=True)

    clock.current = float(len(connections) - 1 + advance)
    expired_indexes = {
        connection_index
        for connection_index in range(len(connections))
        if clock.current - connection_index > idle_timeout
    }
    expected_reused_index = next(
        (
            connection_index
            for connection_index in reversed(range(len(connections)))
            if connection_index not in expired_indexes
            and origin_indexes[connection_index] == trigger_origin_index
        ),
        None,
    )

    acquired = await pool.acquire(ORIGINS[trigger_origin_index])

    if expected_reused_index is None:
        assert acquired not in connections
        assert opener.dial_count == len(connections) + 1
    else:
        assert acquired is connections[expected_reused_index]
        assert opener.dial_count == len(connections)
    for connection_index, writer in enumerate(
        opener.writers[: len(connections)]
    ):
        if connection_index in expired_indexes:
            writer.close.assert_called_once()
        else:
            writer.close.assert_not_called()

    await pool.release(acquired, reusable=False)
    await pool.aclose()

    for writer in opener.writers:
        writer.close.assert_called_once()


@pytest.mark.asyncio
@given(
    origin_indexes=st.lists(
        st.integers(min_value=0, max_value=len(ORIGINS) - 1),
        min_size=1,
        max_size=8,
    ),
    idle_timeout=st.integers(min_value=0, max_value=12),
    advance=st.integers(min_value=0, max_value=12),
    trigger_origin_index=st.integers(
        min_value=0,
        max_value=len(ORIGINS) - 1,
    ),
)
async def test_pool_expires_exactly_connections_past_timeout(
    origin_indexes: list[int],
    idle_timeout: int,
    advance: int,
    trigger_origin_index: int,
) -> None:
    await _check_expiration_history(
        origin_indexes,
        idle_timeout,
        advance,
        trigger_origin_index,
    )


def test_origin_from_uri_uses_scheme_host_port() -> None:
    uri = ParsedURI(scheme="https", host="example.com", port=8443, path="/x")

    origin = Origin.from_uri(uri)

    assert origin == Origin(scheme="https", host="example.com", port=8443)


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_max_connections_per_origin_raises_value_error(
    limit: int,
) -> None:
    with pytest.raises(ValueError, match="must be at least 1"):
        ConnectionPool(FakeOpener(), max_connections_per_origin=limit)


@pytest.mark.asyncio
async def test_is_closed_false_on_live_streams() -> None:
    connection = _make_connection(asyncio.StreamReader())

    assert not connection.is_closed()


@pytest.mark.asyncio
async def test_is_closed_true_after_reader_eof() -> None:
    reader = asyncio.StreamReader()
    reader.feed_eof()

    assert _make_connection(reader).is_closed()


@pytest.mark.asyncio
async def test_is_closed_true_when_writer_is_closing() -> None:
    connection = _make_connection(asyncio.StreamReader())
    connection.close()

    assert connection.is_closed()


@pytest.mark.asyncio
async def test_has_idle_bytes_false_on_empty_buffer() -> None:
    connection = _make_connection(asyncio.StreamReader())

    assert not await connection.has_idle_bytes()


@pytest.mark.asyncio
async def test_has_idle_bytes_true_when_data_buffered() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"x")

    assert await _make_connection(reader).has_idle_bytes()


@pytest.mark.asyncio
async def test_has_idle_bytes_false_on_quiet_socket() -> None:
    async with _socket_connection() as (connection, _):
        assert not await connection.has_idle_bytes()


@pytest.mark.asyncio
async def test_has_idle_bytes_true_when_data_pending_in_kernel() -> None:
    async with _socket_connection() as (connection, peer):
        # Sent through the raw peer socket with no event loop
        # iteration in between, so the bytes are in the kernel but
        # not yet in the StreamReader buffer.
        peer.send(b"x")

        assert await connection.has_idle_bytes()


@pytest.mark.asyncio
async def test_has_idle_bytes_true_when_eof_pending_in_kernel() -> None:
    async with _socket_connection() as (connection, peer):
        peer.close()

        assert not connection.is_closed()
        assert await connection.has_idle_bytes()


@pytest.mark.asyncio
async def test_has_idle_bytes_true_after_transport_closed() -> None:
    async with _socket_connection() as (connection, _):
        connection.close()
        await connection.wait_closed()

        assert await connection.has_idle_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("disturb", [_send_stray_byte, _close_peer])
async def test_acquire_replaces_idle_connection_with_pending_input(
    disturb: Callable[[socket.socket], None],
) -> None:
    opener = SocketPairOpener()
    pool = ConnectionPool(opener, idle_timeout=None)
    try:
        first = await pool.acquire(ORIGIN_A)
        await pool.release(first, reusable=True)
        disturb(opener.peers[0])

        second = await pool.acquire(ORIGIN_A)

        assert second is not first
        assert first.is_closed()
        assert opener.dial_count == 2
        await pool.release(second, reusable=True)
    finally:
        await pool.aclose()
        opener.close()


@pytest.mark.asyncio
async def test_acquire_dials_new_connection() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)

    connection = await pool.acquire(ORIGIN_A)

    assert connection.origin == ORIGIN_A
    assert opener.dial_count == 1
    await pool.release(connection, reusable=True)
    await pool.aclose()


@pytest.mark.asyncio
async def test_reusable_release_is_reused_on_next_acquire() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    first = await pool.acquire(ORIGIN_A)
    await pool.release(first, reusable=True)

    second = await pool.acquire(ORIGIN_A)

    assert second is first
    assert opener.dial_count == 1
    await pool.release(second, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_checkout_reuses_most_recently_released_connection() -> None:
    opener = FakeOpener()
    clock = FakeClock()
    pool = ConnectionPool(opener, clock=clock)
    first = await pool.acquire(ORIGIN_A)
    second = await pool.acquire(ORIGIN_A)
    await pool.release(first, reusable=True)
    clock.current = 1.0
    await pool.release(second, reusable=True)

    reused = await pool.acquire(ORIGIN_A)

    assert reused is second
    await pool.release(reused, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_non_reusable_release_closes_connection() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    connection = await pool.acquire(ORIGIN_A)

    await pool.release(connection, reusable=False)

    opener.writers[0].close.assert_called_once()
    fresh = await pool.acquire(ORIGIN_A)
    assert fresh is not connection
    assert opener.dial_count == 2
    await pool.release(fresh, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_zero_idle_cap_closes_every_reusable_release() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener, max_idle_connections=0)
    connection = await pool.acquire(ORIGIN_A)

    await pool.release(connection, reusable=True)

    opener.writers[0].close.assert_called_once()
    await pool.aclose()


@pytest.mark.asyncio
async def test_acquire_waits_for_slot_at_per_origin_cap() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener, max_connections_per_origin=1)
    held = await pool.acquire(ORIGIN_A)
    waiter = asyncio.create_task(pool.acquire(ORIGIN_A))
    await asyncio.sleep(0)
    assert not waiter.done()

    await pool.release(held, reusable=True)

    reused = await asyncio.wait_for(waiter, timeout=1.0)
    assert reused is held
    await pool.release(reused, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_cancelled_acquirer_stops_waiting_for_slot() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener, max_connections_per_origin=1)
    held = await pool.acquire(ORIGIN_A)
    waiter = asyncio.create_task(pool.acquire(ORIGIN_A))
    await asyncio.sleep(0)

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    await pool.release(held, reusable=False)
    fresh = await asyncio.wait_for(pool.acquire(ORIGIN_A), timeout=1.0)
    await pool.release(fresh, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_acquire_prunes_expired_idle_across_origins() -> None:
    opener = FakeOpener()
    clock = FakeClock()
    pool = ConnectionPool(opener, idle_timeout=5.0, clock=clock)
    conn_a = await pool.acquire(ORIGIN_A)
    conn_b = await pool.acquire(ORIGIN_B)
    await pool.release(conn_a, reusable=True)
    await pool.release(conn_b, reusable=True)
    clock.current = 6.0

    fresh = await pool.acquire(ORIGIN_A)

    assert fresh is not conn_a
    assert opener.dial_count == 3
    opener.writers[0].close.assert_called_once()
    opener.writers[1].close.assert_called_once()
    await pool.release(fresh, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_prune_keeps_unexpired_idle_connections() -> None:
    opener = FakeOpener()
    clock = FakeClock()
    pool = ConnectionPool(opener, idle_timeout=5.0, clock=clock)
    old = await pool.acquire(ORIGIN_A)
    fresh = await pool.acquire(ORIGIN_A)
    await pool.release(old, reusable=True)
    clock.current = 4.0
    await pool.release(fresh, reusable=True)
    clock.current = 6.0

    reused = await pool.acquire(ORIGIN_A)

    assert reused is fresh
    opener.writers[0].close.assert_called_once()
    await pool.release(reused, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_none_idle_timeout_disables_expiry() -> None:
    opener = FakeOpener()
    clock = FakeClock()
    pool = ConnectionPool(opener, idle_timeout=None, clock=clock)
    connection = await pool.acquire(ORIGIN_A)
    await pool.release(connection, reusable=True)
    clock.current = 1e9

    reused = await pool.acquire(ORIGIN_A)

    assert reused is connection
    await pool.release(reused, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_checkout_discards_connection_with_idle_bytes() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    connection = await pool.acquire(ORIGIN_A)
    await pool.release(connection, reusable=True)
    opener.readers[0].feed_data(b"HTTP/1.1 408 Request Timeout\r\n\r\n")

    fresh = await pool.acquire(ORIGIN_A)

    assert fresh is not connection
    assert opener.dial_count == 2
    opener.writers[0].close.assert_called_once()
    await pool.release(fresh, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_checkout_discards_connection_closed_while_idle() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    connection = await pool.acquire(ORIGIN_A)
    await pool.release(connection, reusable=True)
    opener.readers[0].feed_eof()

    fresh = await pool.acquire(ORIGIN_A)

    assert fresh is not connection
    assert opener.dial_count == 2
    opener.writers[0].close.assert_called_once()
    await pool.release(fresh, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_checkout_skips_dead_connection_and_returns_next() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    first = await pool.acquire(ORIGIN_A)
    second = await pool.acquire(ORIGIN_A)
    await pool.release(first, reusable=True)
    await pool.release(second, reusable=True)
    opener.readers[1].feed_eof()

    reused = await pool.acquire(ORIGIN_A)

    assert reused is first
    opener.writers[1].close.assert_called_once()
    await pool.release(reused, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_release_evicts_oldest_idle_connection_globally() -> None:
    opener = FakeOpener()
    clock = FakeClock()
    pool = ConnectionPool(opener, max_idle_connections=1, clock=clock)
    conn_a = await pool.acquire(ORIGIN_A)
    conn_b = await pool.acquire(ORIGIN_B)
    await pool.release(conn_a, reusable=True)
    clock.current = 1.0

    await pool.release(conn_b, reusable=True)

    opener.writers[0].close.assert_called_once()
    reused = await pool.acquire(ORIGIN_B)
    assert reused is conn_b
    await pool.release(reused, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_release_evicts_idle_connection_from_same_origin() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener, max_idle_connections=1)
    first = await pool.acquire(ORIGIN_A)
    second = await pool.acquire(ORIGIN_A)
    await pool.release(first, reusable=True)

    await pool.release(second, reusable=True)

    opener.writers[0].close.assert_called_once()
    reused = await pool.acquire(ORIGIN_A)
    assert reused is second
    await pool.release(reused, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_eviction_frees_capacity_on_evicted_origin() -> None:
    opener = FakeOpener()
    clock = FakeClock()
    pool = ConnectionPool(
        opener,
        max_connections_per_origin=1,
        max_idle_connections=1,
        clock=clock,
    )
    conn_a = await pool.acquire(ORIGIN_A)
    await pool.release(conn_a, reusable=True)
    conn_b = await pool.acquire(ORIGIN_B)
    clock.current = 1.0
    await pool.release(conn_b, reusable=True)

    fresh_a = await asyncio.wait_for(pool.acquire(ORIGIN_A), timeout=1.0)

    assert fresh_a is not conn_a
    assert opener.dial_count == 3
    await pool.release(fresh_a, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_dial_failure_releases_reserved_slot() -> None:
    attempts = 0

    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("connection refused")
        return _make_stream_pair()

    pool = ConnectionPool(opener, max_connections_per_origin=1)
    with pytest.raises(OSError):
        await pool.acquire(ORIGIN_A)

    connection = await asyncio.wait_for(pool.acquire(ORIGIN_A), timeout=1.0)

    await pool.release(connection, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_dial_failure_wakes_waiter_blocked_on_slot() -> None:
    started = asyncio.Event()
    proceed = asyncio.Event()
    attempts = 0

    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            started.set()
            await proceed.wait()
            raise OSError("connection refused")
        return _make_stream_pair()

    pool = ConnectionPool(opener, max_connections_per_origin=1)
    failing = asyncio.create_task(pool.acquire(ORIGIN_A))
    await started.wait()
    waiter = asyncio.create_task(pool.acquire(ORIGIN_A))
    await asyncio.sleep(0)
    assert not waiter.done()
    proceed.set()

    with pytest.raises(OSError):
        await failing
    connection = await asyncio.wait_for(waiter, timeout=1.0)

    await pool.release(connection, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_cancelled_dial_releases_reserved_slot() -> None:
    started = asyncio.Event()
    never = asyncio.Event()
    attempts = 0

    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            started.set()
            await never.wait()
        return _make_stream_pair()

    pool = ConnectionPool(opener, max_connections_per_origin=1)
    task = asyncio.create_task(pool.acquire(ORIGIN_A))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    connection = await asyncio.wait_for(pool.acquire(ORIGIN_A), timeout=1.0)

    await pool.release(connection, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_parser_factory_failure_closes_dial_and_releases_slot() -> None:
    failed_writer = _make_writer()
    failed_writer.close.side_effect = OSError("close failed")
    failed_writer.wait_closed = AsyncMock(side_effect=OSError("drain failed"))
    parser_attempts = 0
    opener_attempts = 0

    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        nonlocal opener_attempts
        opener_attempts += 1
        if opener_attempts == 1:
            return asyncio.StreamReader(), failed_writer
        return _make_stream_pair()

    def parser_factory() -> AsyncMultiResponseParser:
        nonlocal parser_attempts
        parser_attempts += 1
        if parser_attempts == 1:
            raise ValueError("parser construction failed")
        return AsyncMultiResponseParser()

    pool = ConnectionPool(
        opener,
        max_connections_per_origin=1,
        parser_factory=parser_factory,
    )

    with pytest.raises(ValueError):
        await pool.acquire(ORIGIN_A)

    failed_writer.close.assert_called_once()
    connection = await asyncio.wait_for(pool.acquire(ORIGIN_A), timeout=1.0)
    await pool.release(connection, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_cancelled_post_dial_lock_wait_closes_connection() -> None:
    blocking_reader = CancellationResistantReader()
    reader_a, writer_a = _make_stream_pair()
    writer_b = _make_writer()
    dial_started = asyncio.Event()
    dial_can_return = asyncio.Event()
    a_attempts = 0
    b_attempts = 0

    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        nonlocal a_attempts, b_attempts
        if origin == ORIGIN_B:
            b_attempts += 1
            if b_attempts == 1:
                return blocking_reader, writer_b
            return _make_stream_pair()
        a_attempts += 1
        if a_attempts == 1:
            dial_started.set()
            await dial_can_return.wait()
            return reader_a, writer_a
        return _make_stream_pair()

    pool = ConnectionPool(opener, max_connections_per_origin=1)
    idle_b = await pool.acquire(ORIGIN_B)
    await pool.release(idle_b, reusable=True)
    dial = asyncio.create_task(pool.acquire(ORIGIN_A))
    await dial_started.wait()
    lock_holder = asyncio.create_task(pool.acquire(ORIGIN_B))
    await blocking_reader.started.wait()
    dial_can_return.set()
    await asyncio.sleep(0)

    dial.cancel()
    await asyncio.sleep(0)
    dial.cancel()
    await asyncio.sleep(0)

    assert not dial.done()
    blocking_reader.proceed.set()
    held_b = await asyncio.wait_for(lock_holder, timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await dial
    writer_a.close.assert_called_once()

    fresh_a = await asyncio.wait_for(pool.acquire(ORIGIN_A), timeout=1.0)
    await pool.release(held_b, reusable=False)
    await pool.release(fresh_a, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_cancelled_release_finishes_before_propagating() -> None:
    blocking_reader = CancellationResistantReader()
    first_writer = _make_writer()
    opener_calls = 0

    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        nonlocal opener_calls
        opener_calls += 1
        if opener_calls == 1:
            return blocking_reader, first_writer
        return _make_stream_pair()

    pool = ConnectionPool(opener, max_connections_per_origin=2)
    first = await pool.acquire(ORIGIN_A)
    second = await pool.acquire(ORIGIN_A)
    second_writer = second.writer
    await pool.release(first, reusable=True)
    lock_holder = asyncio.create_task(pool.acquire(ORIGIN_A))
    await blocking_reader.started.wait()
    release = asyncio.create_task(pool.release(second, reusable=False))
    await asyncio.sleep(0)

    release.cancel()
    await asyncio.sleep(0)

    assert not release.done()
    blocking_reader.proceed.set()
    held = await asyncio.wait_for(lock_holder, timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await release
    second_writer.close.assert_called_once()

    fresh = await asyncio.wait_for(pool.acquire(ORIGIN_A), timeout=1.0)
    await pool.release(held, reusable=False)
    await pool.release(fresh, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_probe_error_closes_popped_connection() -> None:
    reader = Mock(spec=asyncio.StreamReader)
    reader.at_eof.return_value = False
    reader.read = AsyncMock(side_effect=ValueError("broken reader"))
    writer = _make_writer()

    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return reader, writer

    pool = ConnectionPool(opener, max_connections_per_origin=1)
    connection = await pool.acquire(ORIGIN_A)
    await pool.release(connection, reusable=True)

    with pytest.raises(ValueError):
        await pool.acquire(ORIGIN_A)

    writer.close.assert_called_once()
    await pool.aclose()


@pytest.mark.asyncio
async def test_close_error_does_not_break_release() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    connection = await pool.acquire(ORIGIN_A)
    opener.writers[0].close.side_effect = OSError("already dead")

    await pool.release(connection, reusable=False)

    fresh = await asyncio.wait_for(pool.acquire(ORIGIN_A), timeout=1.0)
    await pool.release(fresh, reusable=False)
    await pool.aclose()


@pytest.mark.asyncio
async def test_drain_swallows_os_error_from_wait_closed() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    connection = await pool.acquire(ORIGIN_A)
    opener.writers[0].wait_closed = AsyncMock(side_effect=OSError("reset"))

    await pool.release(connection, reusable=False)

    await pool.aclose()


@pytest.mark.asyncio
async def test_pool_usable_while_transport_shutdown_hangs() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    connection = await pool.acquire(ORIGIN_A)
    hanging = asyncio.Event()
    opener.writers[0].wait_closed = AsyncMock(side_effect=hanging.wait)
    await pool.release(connection, reusable=False)

    fresh = await asyncio.wait_for(pool.acquire(ORIGIN_A), timeout=1.0)

    assert fresh is not connection
    await pool.release(fresh, reusable=False)
    hanging.set()
    await pool.aclose()


@pytest.mark.asyncio
async def test_acquire_after_aclose_raises_runtime_error() -> None:
    pool = ConnectionPool(FakeOpener())
    await pool.aclose()

    with pytest.raises(RuntimeError):
        await pool.acquire(ORIGIN_A)


@pytest.mark.asyncio
async def test_aclose_closes_idle_connections() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    connection = await pool.acquire(ORIGIN_A)
    await pool.release(connection, reusable=True)

    await pool.aclose()

    opener.writers[0].close.assert_called_once()


@pytest.mark.asyncio
async def test_cancelled_aclose_closes_pool_before_propagating() -> None:
    blocking_reader = CancellationResistantReader()
    idle_writer = _make_writer()
    readers: list[asyncio.StreamReader] = [blocking_reader]

    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        if origin == ORIGIN_B:
            return asyncio.StreamReader(), idle_writer
        reader = readers.pop() if readers else asyncio.StreamReader()
        return reader, _make_writer()

    pool = ConnectionPool(opener)
    probed = await pool.acquire(ORIGIN_A)
    await pool.release(probed, reusable=True)
    idle = await pool.acquire(ORIGIN_B)
    await pool.release(idle, reusable=True)
    lock_holder = asyncio.create_task(pool.acquire(ORIGIN_A))
    await blocking_reader.started.wait()
    closer = asyncio.create_task(pool.aclose())
    await asyncio.sleep(0)

    closer.cancel()
    await asyncio.sleep(0)

    assert not closer.done()
    blocking_reader.proceed.set()
    with pytest.raises(RuntimeError):
        await asyncio.wait_for(lock_holder, timeout=1.0)
    with pytest.raises(asyncio.CancelledError):
        await closer
    idle_writer.close.assert_called_once()
    with pytest.raises(RuntimeError):
        await pool.acquire(ORIGIN_B)


@pytest.mark.asyncio
async def test_aclose_wakes_blocked_acquirer() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener, max_connections_per_origin=1)
    held = await pool.acquire(ORIGIN_A)
    waiter = asyncio.create_task(pool.acquire(ORIGIN_A))
    await asyncio.sleep(0)
    assert not waiter.done()

    await pool.aclose()

    with pytest.raises(RuntimeError):
        await waiter
    await pool.release(held, reusable=True)
    opener.writers[0].close.assert_called_once()
    await pool.aclose()


@pytest.mark.asyncio
async def test_aclose_awaits_drains_from_earlier_closes() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    connection = await pool.acquire(ORIGIN_A)
    finish_close = asyncio.Event()
    opener.writers[0].wait_closed = AsyncMock(side_effect=finish_close.wait)
    await pool.release(connection, reusable=False)

    aclose_task = asyncio.create_task(pool.aclose())
    await asyncio.sleep(0)
    assert not aclose_task.done()

    finish_close.set()
    await asyncio.wait_for(aclose_task, timeout=1.0)


@pytest.mark.asyncio
async def test_cancelled_aclose_keeps_pending_drains_alive() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    connection = await pool.acquire(ORIGIN_A)
    drain_started = asyncio.Event()
    finish_close = asyncio.Event()

    async def wait_closed() -> None:
        drain_started.set()
        await finish_close.wait()

    opener.writers[0].wait_closed = AsyncMock(side_effect=wait_closed)
    await pool.release(connection, reusable=False)
    await drain_started.wait()
    first_close = asyncio.create_task(pool.aclose())
    await asyncio.sleep(0)

    first_close.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_close

    second_close = asyncio.create_task(pool.aclose())
    await asyncio.sleep(0)
    assert not second_close.done()

    finish_close.set()
    await asyncio.wait_for(second_close, timeout=1.0)


@pytest.mark.asyncio
async def test_aclose_racing_dial_closes_fresh_connection() -> None:
    started = asyncio.Event()
    proceed = asyncio.Event()
    reader, writer = _make_stream_pair()

    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        started.set()
        await proceed.wait()
        return reader, writer

    pool = ConnectionPool(opener)
    task = asyncio.create_task(pool.acquire(ORIGIN_A))
    await started.wait()
    await pool.aclose()
    proceed.set()

    with pytest.raises(RuntimeError):
        await task

    writer.close.assert_called_once()
    await pool.aclose()


@pytest.mark.asyncio
async def test_failed_dial_does_not_retain_origin() -> None:
    async def opener(
        origin: Origin,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        raise OSError("connection refused")

    pool = ConnectionPool(opener)
    origin = Origin(scheme="http", host="one-off.example", port=80)
    reference = weakref.ref(origin)

    with pytest.raises(OSError):
        await pool.acquire(origin)

    del origin
    gc.collect()
    assert reference() is None
    await pool.aclose()


@pytest.mark.asyncio
async def test_closed_connection_does_not_retain_origin() -> None:
    opener = FakeOpener()
    pool = ConnectionPool(opener)
    origin = Origin(scheme="http", host="one-off.example", port=80)
    reference = weakref.ref(origin)
    connection = await pool.acquire(origin)

    await pool.release(connection, reusable=False)
    await pool.aclose()
    del connection, origin
    await asyncio.sleep(0)
    gc.collect()

    assert reference() is None
