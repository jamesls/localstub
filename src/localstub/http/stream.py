"""Stream-level read-ahead for asyncio stream readers.

A parser working through a stream can read past the end of the
current message: the surplus bytes belong to the next consumer of
the stream (a pipelined request, or another protocol after an
Upgrade or CONNECT).  ``asyncio.StreamReader`` offers no way to put
bytes back at the front of its buffer (``feed_data()`` appends after
already-buffered data and fails after EOF), so this module holds
them keyed by the reader they came from.

Any consumer that reads through :func:`read` sees the stream's bytes
in their original order, whether they were held here or are still
buffered in the reader.  A consumer that reads from the reader
directly should first drain :func:`take_unread_data`.
"""

from __future__ import annotations

import asyncio
from weakref import WeakKeyDictionary

_UNREAD_BY_READER: WeakKeyDictionary[asyncio.StreamReader, bytearray] = (
    WeakKeyDictionary()
)


def unread_data(
    reader: asyncio.StreamReader,
    data: bytes | bytearray,
) -> None:
    """Return ``data`` to ``reader``, ahead of any future reads.

    The bytes are handed back out before anything still buffered in
    the reader itself, preserving stream order.  Unread bytes are
    held for the lifetime of the reader object.
    """
    if not data:
        return
    buffer = _UNREAD_BY_READER.setdefault(reader, bytearray())
    buffer[:0] = data


def take_unread_data(
    reader: asyncio.StreamReader,
    limit: int | None = None,
) -> bytes:
    """Consume bytes previously returned to ``reader`` via unread_data().

    Returns up to ``limit`` bytes, or everything held when ``limit``
    is None.  Returns ``b""`` when nothing is held.  Call this before
    handing the stream to a consumer that reads from the reader
    directly instead of through :func:`read`.
    """
    buffer = _UNREAD_BY_READER.get(reader)
    if buffer is None:
        return b""
    size = len(buffer) if limit is None else min(limit, len(buffer))
    data = bytes(buffer[:size])
    del buffer[:size]
    if not buffer:
        del _UNREAD_BY_READER[reader]
    return data


async def read(reader: asyncio.StreamReader, max_bytes: int) -> bytes:
    """Read up to ``max_bytes`` from ``reader``, unread bytes first.

    Drop-in replacement for ``reader.read(max_bytes)`` that yields
    bytes returned via :func:`unread_data` before reading from the
    stream itself.
    """
    data = take_unread_data(reader, max_bytes)
    if data:
        return data
    return await reader.read(max_bytes)
