"""Bounded buffers for recorded HTTP traffic.

Recording consumers (tests, the CLI) usually drain records as traffic
flows.  When a consumer never drains a stream, an unbounded buffer would
retain every request and response body for the lifetime of the process.
The buffers here evict their oldest entries past a configured capacity
so memory stays bounded no matter how long the server or proxy runs.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Final

LOG = logging.getLogger(__name__)

DEFAULT_RECORDING_BUFFER_SIZE: Final = 256


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
        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
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
