"""Connection pooling for the built-in asyncio HTTP client.

The pool keys idle upstream connections by origin, hands each one to
at most one exchange at a time, and closes a connection whenever its
byte-stream position is in doubt: closing is always RFC-correct and
costs one reconnect, while reusing a desynced connection corrupts
every later response on it.
"""

from __future__ import annotations

import asyncio
import logging
import selectors
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

from localstub.clock import Clock, MonotonicClock
from localstub.http.response import AsyncMultiResponseParser
from localstub.http.uri import ParsedURI

LOG = logging.getLogger(__name__)

DEFAULT_MAX_CONNECTIONS_PER_ORIGIN = 10
DEFAULT_MAX_IDLE_CONNECTIONS = 20
DEFAULT_IDLE_TIMEOUT = 5.0


@dataclass(frozen=True)
class Origin:
    """Scheme/host/port triple identifying an upstream server."""

    scheme: str
    host: str
    port: int

    @classmethod
    def from_uri(cls, uri: ParsedURI) -> Origin:
        return cls(scheme=uri.scheme, host=uri.host, port=uri.port)


type ConnectionOpener = Callable[
    [Origin],
    Awaitable[tuple[asyncio.StreamReader, asyncio.StreamWriter]],
]

type ParserFactory = Callable[[], AsyncMultiResponseParser]


class _SocketLike(Protocol):
    """The part of a transport's ``socket`` extra the probe needs."""

    def fileno(self) -> int: ...


def _socket_is_readable(sock: _SocketLike) -> bool:
    """Whether the kernel holds unread bytes or a pending EOF.

    A closed socket has no descriptor to poll; the stream-level
    probe reports its EOF instead.
    """
    fd = sock.fileno()
    if fd < 0:
        return False
    with selectors.DefaultSelector() as selector:
        selector.register(fd, selectors.EVENT_READ)
        return bool(selector.select(timeout=0))


class PooledConnection:
    """One upstream connection plus the parser that owns its stream.

    The parser travels with the connection because its buffer can hold
    bytes read past the end of one response; a fresh parser per
    exchange would silently drop them.
    """

    def __init__(
        self,
        origin: Origin,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        parser: AsyncMultiResponseParser,
    ) -> None:
        self.origin = origin
        self.reader = reader
        self.writer = writer
        self.parser = parser
        self.idle_since: float = 0.0

    def is_closed(self) -> bool:
        """Whether the transport is known dead (EOF seen or closing)."""
        return self.reader.at_eof() or self.writer.is_closing()

    async def has_idle_bytes(self) -> bool:
        """Whether bytes or EOF arrived while the connection sat idle.

        Idle input can sit in the kernel socket or in the
        ``StreamReader`` buffer, and both are checked without waiting
        on the network.  The socket is polled first and synchronously:
        the event loop moves socket input into the reader only on a
        later iteration, behind callbacks already queued, so a probe
        that suspends before looking at the socket loses that race
        and reports a connection with input already pending as clean.

        The reader is checked next.  A non-empty buffer satisfies the
        read synchronously, while an empty buffer suspends the read
        and the already-expired timeout cancels it.  Consuming one
        byte is harmless because a flagged connection is closed,
        never reused.
        """
        sock: _SocketLike | None = self.writer.get_extra_info("socket")
        if sock is not None and _socket_is_readable(sock):
            return True
        try:
            async with asyncio.timeout(0):
                await self.reader.read(1)
        except TimeoutError:
            return False
        return True

    def close(self) -> None:
        """Start shutting the transport down."""
        self.writer.close()

    async def wait_closed(self) -> None:
        """Wait until the transport has finished closing."""
        await self.writer.wait_closed()


class _OriginState:
    """Per-origin bookkeeping, guarded by the pool's shared lock.

    ``idle`` is a stack ordered by ``idle_since``: oldest first,
    newest last, so checkout pops the connection least likely to have
    hit the server's keep-alive timeout.  ``open_count`` counts idle
    plus in-flight connections.  The condition shares the pool lock so
    a release on one origin can update another origin's state and
    notify its waiters in one critical section.
    """

    def __init__(self, lock: asyncio.Lock) -> None:
        self.idle: list[PooledConnection] = []
        self.open_count = 0
        self.waiter_count = 0
        self.condition = asyncio.Condition(lock)


class ConnectionPool:
    """Owns idle upstream connections; one exchange per connection.

    A connection is either idle in the pool or held by exactly one
    caller between ``acquire()`` and ``release()``.  All state is
    guarded by one lock; only dials and shutdown drains wait on the
    network, and neither runs while the lock is held.
    """

    def __init__(
        self,
        opener: ConnectionOpener,
        *,
        max_connections_per_origin: int = DEFAULT_MAX_CONNECTIONS_PER_ORIGIN,
        max_idle_connections: int = DEFAULT_MAX_IDLE_CONNECTIONS,
        idle_timeout: float | None = DEFAULT_IDLE_TIMEOUT,
        clock: Clock | None = None,
        parser_factory: ParserFactory | None = None,
    ) -> None:
        if max_connections_per_origin < 1:
            raise ValueError(
                "max_connections_per_origin must be at least 1, "
                f"got {max_connections_per_origin}"
            )
        self._opener = opener
        self._max_connections_per_origin = max_connections_per_origin
        self._max_idle_connections = max_idle_connections
        self._idle_timeout = idle_timeout
        self._clock = clock if clock is not None else MonotonicClock()
        self._parser_factory = (
            parser_factory
            if parser_factory is not None
            else AsyncMultiResponseParser
        )
        self._lock = asyncio.Lock()
        self._origins: dict[Origin, _OriginState] = {}
        self._closed = False
        self._drains: set[asyncio.Task[None]] = set()

    async def acquire(self, origin: Origin) -> PooledConnection:
        """Return a healthy connection to *origin*, dialing if needed.

        Raises ``RuntimeError`` once the pool is closed.  When the
        origin is at its connection cap, waits until a slot frees; the
        caller bounds the wait with its own timeout.
        """
        async with self._lock:
            if self._closed:
                raise RuntimeError("connection pool is closed")
            state = self._origin_state(origin)
            while True:
                if self._closed:
                    self._retire_origin(origin, state)
                    raise RuntimeError("connection pool is closed")
                self._prune_expired(retained_state=state)
                try:
                    connection = await self._checkout_idle(state)
                except BaseException:
                    self._retire_origin(origin, state)
                    raise
                if connection is not None:
                    return connection
                if state.open_count < self._max_connections_per_origin:
                    # Reserve the slot before dropping the lock to
                    # dial.
                    state.open_count += 1
                    break
                state.waiter_count += 1
                try:
                    await state.condition.wait()
                except BaseException:
                    state.waiter_count -= 1
                    self._retire_origin(origin, state)
                    raise
                state.waiter_count -= 1
        return await self._dial(origin, state)

    async def release(
        self, connection: PooledConnection, *, reusable: bool
    ) -> None:
        """Return *connection* to the idle pool or close it.

        Once called, the bookkeeping transition finishes before any
        cancellation received while waiting for the lock propagates.
        """
        cancellation = await self._acquire_lock_for_cleanup()
        try:
            state = self._origins[connection.origin]
            if not reusable or self._closed or self._max_idle_connections <= 0:
                self._close_connection(connection)
                self._retire_origin(connection.origin, state)
            else:
                if self._idle_count() >= self._max_idle_connections:
                    # The connection just released is fresher and more
                    # likely to survive than the oldest idle one.
                    self._evict_oldest_idle(retained_state=state)
                connection.idle_since = self._clock.now()
                state.idle.append(connection)
                state.condition.notify_all()
        finally:
            self._lock.release()
        if cancellation is not None:
            raise cancellation

    async def aclose(self) -> None:
        """Close the pool and wait for pending transport shutdowns.

        Idle connections close immediately and blocked acquirers are
        woken to raise.  In-flight exchanges are not awaited; their
        connections close on release because the pool no longer
        accepts them.

        A cancellation received while waiting for the lock is
        deferred until the pool is marked closed, so a cancelled
        shutdown never leaves the pool open; it then skips the drain
        wait and propagates, with the transport shutdowns finishing
        in the background.
        """
        cancellation = await self._acquire_lock_for_cleanup()
        try:
            self._closed = True
            for origin, state in list(self._origins.items()):
                for connection in list(state.idle):
                    self._close_connection(connection)
                state.condition.notify_all()
                self._retire_origin(origin, state)
        finally:
            self._lock.release()
        if cancellation is not None:
            raise cancellation
        pending = list(self._drains)
        if pending:
            await asyncio.shield(
                asyncio.gather(*pending, return_exceptions=True)
            )

    def _origin_state(self, origin: Origin) -> _OriginState:
        state = self._origins.get(origin)
        if state is None:
            state = _OriginState(self._lock)
            self._origins[origin] = state
        return state

    def _retire_origin(self, origin: Origin, state: _OriginState) -> None:
        """Forget an origin after its final user and connection leave."""
        if (
            state.open_count == 0
            and state.waiter_count == 0
            and self._origins.get(origin) is state
        ):
            del self._origins[origin]

    async def _acquire_lock_for_cleanup(
        self,
    ) -> asyncio.CancelledError | None:
        """Acquire the pool lock while deferring incoming cancellation."""
        cancellation: asyncio.CancelledError | None = None
        while True:
            try:
                await self._lock.acquire()
            except asyncio.CancelledError as exc:
                cancellation = exc
            else:
                return cancellation

    async def _checkout_idle(
        self, state: _OriginState
    ) -> PooledConnection | None:
        """Pop idle connections newest-first, discarding stale ones."""
        while state.idle:
            connection = state.idle.pop()
            try:
                stale = (
                    connection.is_closed() or await connection.has_idle_bytes()
                )
            except BaseException:
                # Cancellation (or a probe error) must not strand the
                # popped connection outside the pool's bookkeeping.
                self._close_connection(connection)
                raise
            if stale:
                self._close_connection(connection)
                continue
            return connection
        return None

    async def _dial(
        self, origin: Origin, state: _OriginState
    ) -> PooledConnection:
        """Dial *origin* into a slot the caller already reserved."""
        writer: asyncio.StreamWriter | None = None
        connection: PooledConnection | None = None
        pool_closed = False
        try:
            reader, writer = await self._opener(origin)
            connection = PooledConnection(
                origin, reader, writer, self._parser_factory()
            )
            async with self._lock:
                if self._closed:
                    self._close_connection(connection)
                    self._retire_origin(origin, state)
                    pool_closed = True
        except BaseException:
            # Includes CancelledError: any exit that does not produce
            # a connection must release the slot and wake a waiter.
            cancellation = await self._acquire_lock_for_cleanup()
            try:
                if connection is not None:
                    self._close_connection(connection)
                else:
                    state.open_count -= 1
                    state.condition.notify_all()
                    if writer is not None:
                        self._close_writer(writer)
                self._retire_origin(origin, state)
            finally:
                self._lock.release()
            if cancellation is not None:
                raise cancellation
            raise
        if pool_closed:
            raise RuntimeError("connection pool is closed")
        return connection

    def _prune_expired(self, *, retained_state: _OriginState) -> None:
        """Close idle connections past expiry, across all origins.

        Each stack is ordered by ``idle_since``, so expired entries
        cluster at the oldest end and the scan stops at the first
        survivor.
        """
        if self._idle_timeout is None:
            return
        now = self._clock.now()
        for origin, state in list(self._origins.items()):
            while (
                state.idle
                and now - state.idle[0].idle_since > self._idle_timeout
            ):
                self._close_connection(state.idle[0])
            if state is not retained_state:
                self._retire_origin(origin, state)

    def _idle_count(self) -> int:
        return sum(len(state.idle) for state in self._origins.values())

    def _evict_oldest_idle(self, *, retained_state: _OriginState) -> None:
        """Close the oldest idle connection anywhere in the pool.

        Only called when at least one idle connection exists.  The
        evicted connection may belong to another origin; the close
        transition's decrement and notify land there, where a waiter
        can dial into the freed slot.
        """
        states = [
            (origin, state)
            for origin, state in self._origins.items()
            if state.idle
        ]
        origin, oldest = min(
            states, key=lambda item: item[1].idle[0].idle_since
        )
        self._close_connection(oldest.idle[0])
        if oldest is not retained_state:
            self._retire_origin(origin, oldest)

    def _close_connection(self, connection: PooledConnection) -> None:
        """Run the close transition: bookkeeping first, then initiate.

        Synchronous and run under the pool lock, so neither an error
        from the transport nor cancellation can skip the slot release
        or the waiter notify.  The registered drain task lets
        ``aclose()`` await the transport shutdown this close started.
        """
        state = self._origins[connection.origin]
        if connection in state.idle:
            state.idle.remove(connection)
        state.open_count -= 1
        state.condition.notify_all()
        try:
            connection.close()
        except Exception:
            LOG.debug("Failed to close pooled connection", exc_info=True)
        task = asyncio.create_task(self._drain(connection))
        self._drains.add(task)
        task.add_done_callback(self._drains.discard)

    def _close_writer(self, writer: asyncio.StreamWriter) -> None:
        """Initiate and track shutdown for an opened stream writer."""
        try:
            writer.close()
        except Exception:
            LOG.debug("Failed to close pooled connection", exc_info=True)
        task = asyncio.create_task(self._drain_writer(writer))
        self._drains.add(task)
        task.add_done_callback(self._drains.discard)

    async def _drain(self, connection: PooledConnection) -> None:
        try:
            await connection.wait_closed()
        except OSError:
            LOG.debug("Error while draining pooled connection", exc_info=True)

    async def _drain_writer(self, writer: asyncio.StreamWriter) -> None:
        try:
            await writer.wait_closed()
        except OSError:
            LOG.debug("Error while draining pooled connection", exc_info=True)
