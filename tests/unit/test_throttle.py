import pytest

from localstub.http.request import HTTPRequest
from localstub.throttle import (
    MonotonicClock,
    TokenBucket,
    TokenBucketThrottler,
)


class ManualClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now: float = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


def test_token_bucket_allows_burst_then_refills():
    clock = ManualClock()
    bucket = TokenBucket(rate_per_second=2.0, capacity=2.0, clock=clock)

    allowed, retry_after = bucket.try_acquire()
    assert allowed is True
    assert retry_after == 0.0

    allowed, retry_after = bucket.try_acquire()
    assert allowed is True
    assert retry_after == 0.0

    allowed, retry_after = bucket.try_acquire()
    assert allowed is False
    assert retry_after == pytest.approx(0.5)

    clock.advance(0.5)
    allowed, retry_after = bucket.try_acquire()
    assert allowed is True
    assert retry_after == 0.0


def test_monotonic_clock_now_returns_float():
    clock = MonotonicClock()
    first = clock.now()
    second = clock.now()

    assert isinstance(first, float)
    assert second >= first


def test_token_bucket_init_rejects_non_positive_rate():
    clock = ManualClock()
    with pytest.raises(ValueError, match="rate_per_second must be > 0"):
        TokenBucket(rate_per_second=0.0, capacity=1.0, clock=clock)


def test_token_bucket_init_rejects_non_positive_capacity():
    clock = ManualClock()
    with pytest.raises(ValueError, match="capacity must be > 0"):
        TokenBucket(rate_per_second=1.0, capacity=0.0, clock=clock)


def test_token_bucket_try_acquire_rejects_non_positive_amount():
    clock = ManualClock()
    bucket = TokenBucket(rate_per_second=1.0, capacity=1.0, clock=clock)
    with pytest.raises(ValueError, match="amount must be > 0"):
        bucket.try_acquire(amount=0.0)


def test_token_bucket_reset_restores_capacity():
    clock = ManualClock()
    bucket = TokenBucket(rate_per_second=1.0, capacity=1.0, clock=clock)

    assert bucket.try_acquire()[0] is True
    assert bucket.try_acquire()[0] is False

    bucket.reset()
    assert bucket.try_acquire()[0] is True


def test_token_bucket_throttler_default_burst_for_low_rate_is_one():
    clock = ManualClock()
    throttler = TokenBucketThrottler(
        rate_per_second=0.5,
        key=lambda request: "global",
        clock=clock,
    )
    assert throttler.burst == 1.0
    assert throttler.rate_per_second == 0.5

    req = HTTPRequest(method="GET", path="/")
    assert throttler.check(req).allowed is True

    decision = throttler.check(req)
    assert decision.allowed is False
    assert decision.retry_after_seconds == pytest.approx(2.0)


def test_token_bucket_throttler_init_rejects_non_positive_rate():
    clock = ManualClock()
    with pytest.raises(ValueError, match="rate_per_second must be > 0"):
        TokenBucketThrottler(
            rate_per_second=0.0,
            key=lambda request: "global",
            clock=clock,
        )


@pytest.mark.parametrize("burst", [0.0, 0.5])
def test_token_bucket_throttler_init_rejects_burst_below_one(
    burst: float,
):
    clock = ManualClock()
    with pytest.raises(ValueError, match="burst must be >= 1"):
        TokenBucketThrottler(
            rate_per_second=1.0,
            key=lambda request: "global",
            burst=burst,
            clock=clock,
        )


def test_token_bucket_throttler_isolated_per_key():
    clock = ManualClock()
    throttler = TokenBucketThrottler(
        rate_per_second=1.0,
        key=lambda request: request.path or "",
        burst=1.0,
        clock=clock,
    )

    assert throttler.check(HTTPRequest(path="/a")).allowed is True
    assert throttler.check(HTTPRequest(path="/b")).allowed is True

    decision = throttler.check(HTTPRequest(path="/a"))
    assert decision.allowed is False
    assert decision.retry_after_seconds == pytest.approx(1.0)


def test_token_bucket_throttler_reset_clears_state():
    clock = ManualClock()
    throttler = TokenBucketThrottler(
        rate_per_second=1.0,
        key=lambda request: "global",
        burst=1.0,
        clock=clock,
    )

    req = HTTPRequest(path="/")
    assert throttler.check(req).allowed is True
    assert throttler.check(req).allowed is False

    throttler.reset()
    assert throttler.check(req).allowed is True
