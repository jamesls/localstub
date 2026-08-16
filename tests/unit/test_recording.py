from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import pytest

from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HTTPResponse
from localstub.recording import (
    BoundedRecordQueue,
    TrafficRecorder,
    trim_history,
)


class _ManualClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def now(self) -> float:
        return self.value


class _FixedTimestampProvider:
    def __init__(self, value: datetime) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value


def _request(target: str = "/") -> RecordedHTTPRequest:
    request = HTTPRequest(
        method="GET",
        target=target,
        headers=Headers.empty(),
        body=None,
    )
    return RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=f"GET {target} HTTP/1.1\r\n\r\n".encode(),
        http_version="1.1",
    )


def _response() -> RecordedHTTPResponse:
    return RecordedHTTPResponse(
        response=HTTPResponse(status=200),
        reason="OK",
        wire_raw_bytes=b"",
    )


_WALL_TIME = datetime(2026, 8, 16, tzinfo=UTC)


def _recorder(
    buffer_size: int | None = None,
    clock_value: float = 0.0,
) -> TrafficRecorder:
    return TrafficRecorder(
        buffer_size,
        clock=_ManualClock(clock_value),
        timestamp_provider=_FixedTimestampProvider(_WALL_TIME),
    )


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


def test_record_request_returns_injected_timestamps() -> None:
    recorder = _recorder(clock_value=5.5)
    wall, monotonic = recorder.record_request(_request())
    assert wall == _WALL_TIME
    assert monotonic == 5.5


def test_record_request_updates_history_and_last_request() -> None:
    recorder = _recorder()
    first = _request("/a")
    second = _request("/b")
    recorder.record_request(first)
    recorder.record_request(second)
    assert recorder.requests == [first, second]
    assert recorder.last_request is second


def test_record_request_defaults_use_system_clocks() -> None:
    recorder = TrafficRecorder(None)
    wall, monotonic = recorder.record_request(_request())
    assert wall.tzinfo is not None
    assert monotonic > 0


def test_get_request_timestamp_returns_recorded_monotonic_time() -> None:
    clock = _ManualClock(1.0)
    recorder = TrafficRecorder(
        None,
        clock=clock,
        timestamp_provider=_FixedTimestampProvider(_WALL_TIME),
    )
    first = _request("/a")
    recorder.record_request(first)
    clock.value = 2.0
    second = _request("/b")
    recorder.record_request(second)
    assert recorder.get_request_timestamp(first) == 1.0
    assert recorder.get_request_timestamp(second) == 2.0


def test_get_request_timestamp_for_unknown_request_raises_value_error() -> (
    None
):
    recorder = _recorder()
    with pytest.raises(ValueError, match="not found"):
        recorder.get_request_timestamp(_request())


def test_record_request_beyond_buffer_evicts_oldest() -> None:
    recorder = _recorder(buffer_size=2)
    first = _request("/a")
    second = _request("/b")
    third = _request("/c")
    for request in (first, second, third):
        recorder.record_request(request)
    assert recorder.requests == [second, third]
    assert recorder.dropped_requests == 1
    with pytest.raises(ValueError, match="not found"):
        recorder.get_request_timestamp(first)
    assert recorder.get_request_timestamp(second) == 0.0


def test_record_exchange_with_response_records_exchange_and_response() -> None:
    recorder = _recorder()
    request = _request()
    response = _response()
    recorder.record_exchange(
        request=request,
        response=response,
        request_timestamp=_WALL_TIME,
    )
    exchange = recorder.last_exchange
    assert exchange is not None
    assert exchange.request is request
    assert exchange.response is response
    assert exchange.request_timestamp == _WALL_TIME
    assert exchange.response_timestamp == _WALL_TIME
    assert recorder.exchanges == [exchange]
    assert recorder.responses == [response]
    assert recorder.last_response is response
    assert recorder.next_exchange_nowait() is exchange


def test_record_exchange_without_response_skips_response_records() -> None:
    recorder = _recorder()
    recorder.record_exchange(
        request=_request(),
        response=None,
        request_timestamp=_WALL_TIME,
    )
    exchange = recorder.last_exchange
    assert exchange is not None
    assert exchange.response is None
    assert exchange.response_timestamp is None
    assert not recorder.responses
    assert recorder.last_response is None


def test_exchange_and_response_histories_trim_beyond_buffer() -> None:
    recorder = _recorder(buffer_size=2)
    for target in ("/a", "/b", "/c"):
        recorder.record_exchange(
            request=_request(target),
            response=_response(),
            request_timestamp=_WALL_TIME,
        )
    assert len(recorder.exchanges) == 2
    assert len(recorder.responses) == 2
    assert recorder.dropped_exchanges == 1
    assert recorder.dropped_responses == 1


def test_next_exchange_nowait_on_empty_recorder_returns_none() -> None:
    recorder = _recorder()
    assert recorder.next_exchange_nowait() is None


@pytest.mark.asyncio
async def test_next_request_consumes_in_order_and_sets_last_request() -> None:
    recorder = _recorder()
    first = _request("/a")
    second = _request("/b")
    recorder.record_request(first)
    recorder.record_request(second)
    assert recorder.last_request is second
    assert await recorder.next_request() is first
    assert recorder.last_request is first


@pytest.mark.asyncio
async def test_next_request_with_timeout_returns_queued_request() -> None:
    recorder = _recorder()
    request = _request()
    recorder.record_request(request)
    assert await recorder.next_request(timeout=1.0) is request


@pytest.mark.asyncio
async def test_next_request_times_out_when_nothing_recorded() -> None:
    recorder = _recorder()
    with pytest.raises(TimeoutError):
        await recorder.next_request(timeout=0.01)


@pytest.mark.asyncio
async def test_next_response_consumes_in_order_and_sets_last_response() -> (
    None
):
    recorder = _recorder()
    first = _response()
    second = _response()
    for response in (first, second):
        recorder.record_exchange(
            request=_request(),
            response=response,
            request_timestamp=_WALL_TIME,
        )
    assert recorder.last_response is second
    assert await recorder.next_response() is first
    assert recorder.last_response is first


@pytest.mark.asyncio
async def test_next_exchange_returns_completed_exchange() -> None:
    recorder = _recorder()
    recorder.record_exchange(
        request=_request(),
        response=_response(),
        request_timestamp=_WALL_TIME,
    )
    exchange = await recorder.next_exchange()
    assert exchange is recorder.last_exchange


def test_reset_clears_history_queues_and_timestamps() -> None:
    recorder = _recorder(buffer_size=1)
    request = _request()
    recorder.record_request(request)
    recorder.record_request(_request("/b"))
    recorder.record_exchange(
        request=request,
        response=_response(),
        request_timestamp=_WALL_TIME,
    )
    assert recorder.dropped_requests == 1
    recorder.reset()
    assert not recorder.requests
    assert not recorder.responses
    assert not recorder.exchanges
    assert recorder.last_request is None
    assert recorder.last_response is None
    assert recorder.last_exchange is None
    assert recorder.next_exchange_nowait() is None
    assert recorder.dropped_requests == 0
    with pytest.raises(ValueError, match="not found"):
        recorder.get_request_timestamp(request)


def test_reset_preserves_buffer_size() -> None:
    recorder = _recorder(buffer_size=1)
    recorder.reset()
    recorder.record_request(_request("/a"))
    recorder.record_request(_request("/b"))
    assert len(recorder.requests) == 1
    assert recorder.dropped_requests == 1


def test_recorder_with_invalid_buffer_size_raises_value_error() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        TrafficRecorder(0)
