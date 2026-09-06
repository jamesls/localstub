from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from typing import Protocol

from localstub.clock import Clock, MonotonicClock
from localstub.http.request import RecordedHTTPRequest


@dataclass(frozen=True)
class ThrottleDecision:
    allowed: bool
    key: Hashable
    retry_after_seconds: float
    limit_per_second: float


class RequestThrottler(Protocol):
    def check(self, request: RecordedHTTPRequest) -> ThrottleDecision: ...

    def reset(self) -> None: ...


class TokenBucket:
    def __init__(
        self,
        *,
        rate_per_second: float,
        capacity: float,
        clock: Clock,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self._fill_rate: float = rate_per_second
        self._max_capacity: float = capacity
        self._current_capacity: float = capacity
        self._clock: Clock = clock
        self._last_timestamp: float = clock.now()

    def reset(self) -> None:
        self._current_capacity = self._max_capacity
        self._last_timestamp = self._clock.now()

    def try_acquire(self, amount: float = 1.0) -> tuple[bool, float]:
        if amount <= 0:
            raise ValueError("amount must be > 0")
        if amount > self._max_capacity:
            raise ValueError("amount must be <= capacity")
        self._refill()
        if amount <= self._current_capacity:
            self._current_capacity -= amount
            return True, 0.0
        retry_after = (amount - self._current_capacity) / self._fill_rate
        return False, retry_after

    def _refill(self) -> None:
        timestamp = self._clock.now()
        elapsed = timestamp - self._last_timestamp
        if elapsed <= 0:
            return
        fill_amount = elapsed * self._fill_rate
        self._current_capacity = min(
            self._max_capacity,
            self._current_capacity + fill_amount,
        )
        self._last_timestamp = timestamp


ThrottleKeyFunc = Callable[[RecordedHTTPRequest], Hashable]


class TokenBucketThrottler:
    def __init__(
        self,
        *,
        rate_per_second: float,
        key: ThrottleKeyFunc,
        burst: float | None = None,
        clock: Clock | None = None,
    ) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        if burst is not None and burst < 1.0:
            raise ValueError("burst must be >= 1")
        self._rate_per_second: float = rate_per_second
        self._key: ThrottleKeyFunc = key
        if burst is None:
            burst_value = max(1.0, rate_per_second)
        else:
            burst_value = burst
        self._burst: float = burst_value
        self._clock: Clock = clock or MonotonicClock()
        self._buckets: dict[Hashable, TokenBucket] = {}

    @property
    def rate_per_second(self) -> float:
        return self._rate_per_second

    @property
    def burst(self) -> float:
        return self._burst

    def reset(self) -> None:
        self._buckets.clear()

    def check(self, request: RecordedHTTPRequest) -> ThrottleDecision:
        key = self._key(request)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = TokenBucket(
                rate_per_second=self._rate_per_second,
                capacity=self._burst,
                clock=self._clock,
            )
            self._buckets[key] = bucket
        allowed, retry_after = bucket.try_acquire(amount=1.0)
        return ThrottleDecision(
            allowed=allowed,
            key=key,
            retry_after_seconds=retry_after,
            limit_per_second=self._rate_per_second,
        )
