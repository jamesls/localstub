from __future__ import annotations

import asyncio

import pytest

from localstub.recording import BoundedRecordQueue, trim_history


def test_put_and_get_nowait_preserves_fifo_order() -> None:
    queue: BoundedRecordQueue[int] = BoundedRecordQueue(3, name="test")
    queue.put(1)
    queue.put(2)
    assert queue.get_nowait() == 1
    assert queue.get_nowait() == 2


def test_get_nowait_on_empty_queue_returns_none() -> None:
    queue: BoundedRecordQueue[int] = BoundedRecordQueue(1, name="test")
    assert queue.get_nowait() is None


def test_put_beyond_capacity_evicts_oldest() -> None:
    queue: BoundedRecordQueue[int] = BoundedRecordQueue(2, name="test")
    queue.put(1)
    queue.put(2)
    queue.put(3)
    assert queue.dropped == 1
    assert queue.get_nowait() == 2
    assert queue.get_nowait() == 3
    assert queue.get_nowait() is None


def test_dropped_is_zero_when_capacity_not_exceeded() -> None:
    queue: BoundedRecordQueue[int] = BoundedRecordQueue(2, name="test")
    queue.put(1)
    assert queue.dropped == 0


def test_unbounded_queue_never_drops() -> None:
    queue: BoundedRecordQueue[int] = BoundedRecordQueue(None, name="test")
    for i in range(300):
        queue.put(i)
    assert queue.dropped == 0
    assert queue.get_nowait() == 0


def test_zero_maxsize_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BoundedRecordQueue(0, name="test")


def test_negative_maxsize_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BoundedRecordQueue(-5, name="test")


def test_maxsize_property_reports_configured_capacity() -> None:
    queue: BoundedRecordQueue[int] = BoundedRecordQueue(7, name="test")
    assert queue.maxsize == 7


def test_maxsize_property_is_none_for_unbounded() -> None:
    queue: BoundedRecordQueue[int] = BoundedRecordQueue(None, name="test")
    assert queue.maxsize is None


@pytest.mark.asyncio
async def test_get_waits_for_next_put() -> None:
    queue: BoundedRecordQueue[int] = BoundedRecordQueue(2, name="test")
    getter = asyncio.create_task(queue.get())
    await asyncio.sleep(0)
    queue.put(42)
    assert await asyncio.wait_for(getter, timeout=1.0) == 42


@pytest.mark.asyncio
async def test_get_returns_queued_item_immediately() -> None:
    queue: BoundedRecordQueue[int] = BoundedRecordQueue(2, name="test")
    queue.put(7)
    assert await queue.get() == 7


def test_trim_history_with_none_maxsize_keeps_everything() -> None:
    items = list(range(10))
    assert trim_history(items, None) == []
    assert items == list(range(10))


def test_trim_history_under_capacity_keeps_everything() -> None:
    items = [1, 2]
    assert trim_history(items, 5) == []
    assert items == [1, 2]


def test_trim_history_over_capacity_drops_and_returns_oldest() -> None:
    items = [1, 2, 3, 4]
    assert trim_history(items, 2) == [1, 2]
    assert items == [3, 4]
