"""Recording of HTTP traffic into bounded buffers.

Recording consumers (tests, the CLI) usually drain records as traffic
flows.  When a consumer never drains a stream, an unbounded buffer would
retain every request and response body for the lifetime of the process.
The buffers here evict their oldest entries past a configured capacity
so memory stays bounded no matter how long the server or proxy runs.

``TrafficRecorder`` bundles those buffers into the recording subsystem
shared by the test server and the TLS intercept proxy: bounded history
lists, ``last_*`` convenience attributes, and the queues behind the
``next_request()`` / ``next_response()`` / ``next_exchange()`` consumer
API.
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime
from typing import Final

from localstub.clock import Clock, MonotonicClock
from localstub.http.exchange import RecordedExchange
from localstub.http.request import RecordedHTTPRequest
from localstub.http.response import RecordedHTTPResponse
from localstub.middleware import SystemTimestampProvider, TimestampProvider

LOG = logging.getLogger(__name__)

DEFAULT_RECORDING_BUFFER_SIZE: Final = 256
DEFAULT_MAX_CONNECTION_BYTES: Final = 1024 * 1024
DEFAULT_COALESCE_SIZE: Final = 4096


class BoundedByteBuffer:
    """Byte buffer that retains its newest bytes up to a maximum size.

    Retained bytes are held as immutable chunks and joined only when the
    contents are requested.  A write of at least ``coalesce_size`` bytes
    becomes a chunk of its own without being copied, so a large body is
    recorded for the cost of a reference.  Smaller writes are coalesced
    into a pending ``bytearray`` that is sealed into a chunk once it
    reaches ``coalesce_size``, which keeps the chunk count bounded on a
    long connection carrying many small requests.

    Eviction pops whole chunks from the head and records a partial
    eviction of the head chunk as an offset, so staying within
    ``maxsize`` never copies the retained bytes.  It runs when a chunk
    is added or the contents are requested, not on every small write,
    so the buffer may hold fewer than ``coalesce_size`` bytes beyond
    ``maxsize`` in between.  ``len()``, ``dropped`` and ``bytes()``
    always reflect the bounded view.
    """

    def __init__(
        self,
        maxsize: int | None,
        *,
        coalesce_size: int = DEFAULT_COALESCE_SIZE,
    ) -> None:
        if maxsize is not None and maxsize < 1:
            raise ValueError(
                f"maxsize must be at least 1 or None, got {maxsize}"
            )
        if coalesce_size < 1:
            raise ValueError(
                f"coalesce_size must be at least 1, got {coalesce_size}"
            )
        self._maxsize = maxsize
        self._coalesce_size = coalesce_size
        self._chunks: deque[bytes] = deque()
        # Bytes of ``_chunks[0]`` already evicted.  The slice is deferred
        # to ``__bytes__`` so repeated small appends never copy the head.
        self._head_offset = 0
        # Small writes not yet sealed into a chunk.  Always the newest
        # bytes, so it is joined after ``_chunks``.
        self._pending = bytearray()
        # Bytes held, including any not yet evicted past ``maxsize``.
        self._size = 0
        self._dropped = 0

    @property
    def maxsize(self) -> int | None:
        return self._maxsize

    @property
    def dropped(self) -> int:
        """Number of bytes evicted because the buffer was full."""
        return self._dropped + self._overflow()

    def extend(self, data: bytes | bytearray) -> None:
        """Append *data*, evicting the oldest bytes past the maximum."""
        length = len(data)
        if not length:
            return
        maxsize = self._maxsize
        if maxsize is not None and length >= maxsize:
            self._dropped += self._size + length - maxsize
            self._chunks.clear()
            self._chunks.append(bytes(data[-maxsize:]))
            self._head_offset = 0
            self._pending.clear()
            self._size = maxsize
            return

        self._size += length
        if length >= self._coalesce_size:
            self._seal_pending()
            self._chunks.append(bytes(data))
            self._evict_excess()
            return
        self._pending += data
        if len(self._pending) >= self._coalesce_size:
            self._seal_pending()
            self._evict_excess()

    def clear(self) -> None:
        """Remove all retained bytes without counting them as dropped."""
        self._dropped += self._overflow()
        self._chunks.clear()
        self._head_offset = 0
        self._pending.clear()
        self._size = 0

    def __bytes__(self) -> bytes:
        self._evict_excess()
        if self._head_offset:
            self._chunks[0] = self._chunks[0][self._head_offset :]
            self._head_offset = 0
        if not self._pending:
            return b"".join(self._chunks)
        return b"".join((*self._chunks, self._pending))

    def __len__(self) -> int:
        return self._size - self._overflow()

    def _overflow(self) -> int:
        """Bytes past ``maxsize`` whose eviction is still deferred."""
        if self._maxsize is None or self._size <= self._maxsize:
            return 0
        return self._size - self._maxsize

    def _seal_pending(self) -> None:
        if self._pending:
            self._chunks.append(bytes(self._pending))
            self._pending.clear()

    def _evict_excess(self) -> None:
        excess = self._overflow()
        if not excess:
            return
        self._dropped += excess
        self._size -= excess
        chunks = self._chunks
        while excess and chunks:
            retained = len(chunks[0]) - self._head_offset
            if retained > excess:
                self._head_offset += excess
                return
            chunks.popleft()
            self._head_offset = 0
            excess -= retained
        # Only the pending bytes are left once every chunk is gone, which
        # happens when they alone exceed ``maxsize``.  Deleting a prefix
        # of a bytearray advances its start without moving the rest.
        del self._pending[:excess]


class BoundedRecordQueue[T]:
    """FIFO queue that evicts its oldest entry instead of growing forever.

    Unlike ``asyncio.Queue`` with a maxsize, ``put`` never blocks and
    never raises: when the queue is full, the oldest entry is dropped to
    make room and ``dropped`` is incremented.  A ``maxsize`` of ``None``
    means unbounded.
    """

    def __init__(self, maxsize: int | None, name: str) -> None:
        if maxsize is not None and maxsize < 1:
            raise ValueError(
                f"maxsize must be at least 1 or None, got {maxsize}"
            )
        self._maxsize = maxsize
        self._name = name
        self._queue: asyncio.Queue[T] = asyncio.Queue(
            maxsize=0 if maxsize is None else maxsize
        )
        self._dropped = 0

    @property
    def maxsize(self) -> int | None:
        return self._maxsize

    @property
    def dropped(self) -> int:
        """Number of entries evicted because the queue was full."""
        return self._dropped

    def put(self, item: T) -> None:
        """Enqueue *item*, evicting the oldest entry if the queue is full."""
        if self._queue.full():
            self._queue.get_nowait()
            self._dropped += 1
            LOG.debug(
                "%s recording buffer full (maxsize=%s); dropped oldest entry",
                self._name,
                self._maxsize,
            )
        self._queue.put_nowait(item)

    async def get(self) -> T:
        """Wait for and return the oldest entry."""
        return await self._queue.get()

    def get_nowait(self) -> T | None:
        """Return the oldest entry, or ``None`` if the queue is empty."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


def trim_history[T](items: list[T], maxsize: int | None) -> list[T]:
    """Drop the oldest entries so *items* holds at most *maxsize*.

    Returns the evicted entries so callers can release any state
    keyed to them.
    """
    if maxsize is None:
        return []
    excess = len(items) - maxsize
    if excess <= 0:
        return []
    evicted = items[:excess]
    del items[:excess]
    return evicted


class TrafficRecorder:
    """Records HTTP traffic and hands it to recording consumers.

    Owns the bounded request/response/exchange history lists, the
    ``last_*`` convenience attributes, and the queues that back the
    ``next_request()`` / ``next_response()`` / ``next_exchange()``
    consumer API.  All buffers share one ``buffer_size``; ``None``
    means unbounded.
    """

    last_request: RecordedHTTPRequest | None
    requests: list[RecordedHTTPRequest]
    last_response: RecordedHTTPResponse | None
    responses: list[RecordedHTTPResponse]
    last_exchange: RecordedExchange | None
    exchanges: list[RecordedExchange]

    _request_timestamps: dict[int, float]
    _request_queue: BoundedRecordQueue[RecordedHTTPRequest]
    _response_queue: BoundedRecordQueue[RecordedHTTPResponse]
    _exchange_queue: BoundedRecordQueue[RecordedExchange]

    def __init__(
        self,
        buffer_size: int | None = DEFAULT_RECORDING_BUFFER_SIZE,
        *,
        clock: Clock | None = None,
        timestamp_provider: TimestampProvider | None = None,
    ) -> None:
        self._buffer_size = buffer_size
        self._clock: Clock = clock or MonotonicClock()
        self._timestamp_provider: TimestampProvider = (
            timestamp_provider or SystemTimestampProvider()
        )
        self.reset()

    def reset(self) -> None:
        """Drop all recorded traffic and start over with empty buffers."""
        self.last_request = None
        self.requests = []
        self.last_response = None
        self.responses = []
        self.last_exchange = None
        self.exchanges = []
        self._request_timestamps = {}
        self._request_queue = BoundedRecordQueue(
            self._buffer_size, name="requests"
        )
        self._response_queue = BoundedRecordQueue(
            self._buffer_size, name="responses"
        )
        self._exchange_queue = BoundedRecordQueue(
            self._buffer_size, name="exchanges"
        )

    def record_request(
        self, request: RecordedHTTPRequest
    ) -> tuple[datetime, float]:
        """Record *request* and return its reception timestamps.

        Returns a ``(wall_clock, monotonic)`` pair: the wall-clock
        timestamp to later pass to ``record_exchange()`` and the
        monotonic timestamp retrievable via ``get_request_timestamp()``.
        """
        received_monotonic = self._clock.now()
        self.last_request = request
        self.requests.append(request)
        evicted = trim_history(self.requests, self._buffer_size)
        for old in evicted:
            self._request_timestamps.pop(id(old), None)
        self._request_timestamps[id(request)] = received_monotonic
        request_timestamp = self._timestamp_provider.now()
        self._request_queue.put(request)
        return request_timestamp, received_monotonic

    def record_exchange(
        self,
        *,
        request: RecordedHTTPRequest,
        response: RecordedHTTPResponse | None,
        request_timestamp: datetime,
    ) -> None:
        """Record a completed exchange and, when present, its response.

        ``response`` is ``None`` when the exchange failed before a
        response could be sent; the exchange is still recorded so
        consumers observe the request's outcome, but nothing is added
        to the response history or queue.
        """
        response_timestamp = (
            self._timestamp_provider.now() if response is not None else None
        )
        exchange = RecordedExchange(
            request=request,
            response=response,
            request_timestamp=request_timestamp,
            response_timestamp=response_timestamp,
        )
        self.exchanges.append(exchange)
        trim_history(self.exchanges, self._buffer_size)
        self.last_exchange = exchange
        self._exchange_queue.put(exchange)
        if response is not None:
            self.responses.append(response)
            trim_history(self.responses, self._buffer_size)
            self.last_response = response
            self._response_queue.put(response)

    def get_request_timestamp(self, request: RecordedHTTPRequest) -> float:
        """Return the monotonic reception timestamp for *request*.

        Raises:
            ValueError: If the request is not found.  Timestamps are
                only retained for requests still in the ``requests``
                history, which is bounded by ``buffer_size``.
        """
        try:
            return self._request_timestamps[id(request)]
        except KeyError:
            raise ValueError(
                "Request not found in recorded requests"
            ) from None

    async def next_request(
        self, timeout: float | None = None
    ) -> RecordedHTTPRequest:
        """Await and return the next recorded request."""
        request = await self._next(self._request_queue, timeout)
        self.last_request = request
        return request

    async def next_response(
        self, timeout: float | None = None
    ) -> RecordedHTTPResponse:
        """Await and return the next recorded response."""
        response = await self._next(self._response_queue, timeout)
        self.last_response = response
        return response

    async def next_exchange(
        self, timeout: float | None = None
    ) -> RecordedExchange:
        """Await and return the next completed exchange."""
        return await self._next(self._exchange_queue, timeout)

    def next_exchange_nowait(self) -> RecordedExchange | None:
        """Return the next completed exchange, or None if none is queued."""
        return self._exchange_queue.get_nowait()

    @property
    def dropped_requests(self) -> int:
        """Requests evicted unread from the next_request() buffer."""
        return self._request_queue.dropped

    @property
    def dropped_responses(self) -> int:
        """Responses evicted unread from the next_response() buffer."""
        return self._response_queue.dropped

    @property
    def dropped_exchanges(self) -> int:
        """Exchanges evicted unread from the next_exchange() buffer."""
        return self._exchange_queue.dropped

    async def _next[T](
        self, queue: BoundedRecordQueue[T], timeout: float | None
    ) -> T:
        if timeout is None:
            return await queue.get()
        return await asyncio.wait_for(queue.get(), timeout=timeout)
