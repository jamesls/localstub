import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.throttle import (
    MonotonicClock,
    TokenBucket,
    TokenBucketThrottler,
)


def _recorded(method: str, target: str) -> RecordedHTTPRequest:
    request = HTTPRequest(method=method, target=target)
    return RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=f"{method} {target} HTTP/1.1\r\n\r\n".encode(),
        http_version="1.1",
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
    assert allowed
    assert retry_after == pytest.approx(0.0)

    allowed, retry_after = bucket.try_acquire()
    assert allowed
    assert retry_after == pytest.approx(0.0)

    allowed, retry_after = bucket.try_acquire()
    assert not allowed
    assert retry_after == pytest.approx(0.5)

    clock.advance(0.5)
    allowed, retry_after = bucket.try_acquire()
    assert allowed
    assert retry_after == pytest.approx(0.0)


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


def test_token_bucket_try_acquire_rejects_amount_above_capacity() -> None:
    clock = ManualClock()
    bucket = TokenBucket(rate_per_second=1.0, capacity=1.0, clock=clock)
    with pytest.raises(ValueError, match="amount must be <= capacity"):
        bucket.try_acquire(amount=1.5)


def test_token_bucket_reset_restores_capacity():
    clock = ManualClock()
    bucket = TokenBucket(rate_per_second=1.0, capacity=1.0, clock=clock)

    assert bucket.try_acquire()[0]
    assert not bucket.try_acquire()[0]

    bucket.reset()
    assert bucket.try_acquire()[0]


def test_token_bucket_throttler_default_burst_for_low_rate_is_one():
    clock = ManualClock()
    throttler = TokenBucketThrottler(
        rate_per_second=0.5,
        key=lambda request: "global",
        clock=clock,
    )
    assert throttler.burst == pytest.approx(1.0)
    assert throttler.rate_per_second == pytest.approx(0.5)

    req = _recorded("GET", "/")
    assert throttler.check(req).allowed

    decision = throttler.check(req)
    assert not decision.allowed
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
        key=lambda request: request.target,
        burst=1.0,
        clock=clock,
    )

    assert throttler.check(_recorded("GET", "/a")).allowed
    assert throttler.check(_recorded("GET", "/b")).allowed

    decision = throttler.check(_recorded("GET", "/a"))
    assert not decision.allowed
    assert decision.retry_after_seconds == pytest.approx(1.0)


def test_token_bucket_throttler_reset_clears_state():
    clock = ManualClock()
    throttler = TokenBucketThrottler(
        rate_per_second=1.0,
        key=lambda request: "global",
        burst=1.0,
        clock=clock,
    )

    req = _recorded("GET", "/")
    assert throttler.check(req).allowed
    assert not throttler.check(req).allowed

    throttler.reset()
    assert throttler.check(req).allowed


_RETRY_SLACK = 1e-6
_RATES = st.floats(min_value=0.1, max_value=50.0)
_CAPACITIES = st.floats(min_value=0.1, max_value=50.0)
_BURSTS = st.floats(min_value=1.0, max_value=50.0)
_DELTAS = st.floats(min_value=0.0, max_value=100.0)
_AMOUNTS = st.floats(min_value=0.01, max_value=100.0)
_FRACTIONS = st.floats(min_value=0.01, max_value=1.0)
_BucketScenario = tuple[float, float, list[tuple[float, float]]]


@st.composite
def _bucket_scenarios(draw: st.DrawFn) -> _BucketScenario:
    rate = draw(_RATES)
    capacity = draw(_CAPACITIES)
    # Mix absolute amounts with capacity fractions so runs hit both
    # the amount > capacity rejection and the amount == capacity
    # boundary.
    amounts = st.one_of(
        _AMOUNTS,
        _FRACTIONS.map(lambda fraction: capacity * fraction),
    )
    ops = draw(st.lists(st.tuples(_DELTAS, amounts), max_size=20))
    return rate, capacity, ops


def _assert_amount_above_capacity_raises(
    bucket: TokenBucket,
    amount: float,
) -> None:
    with pytest.raises(ValueError, match="amount must be <= capacity"):
        bucket.try_acquire(amount)


@given(scenario=_bucket_scenarios())
def test_token_bucket_denied_acquire_succeeds_after_retry_after(
    scenario: _BucketScenario,
) -> None:
    rate, capacity, ops = scenario
    clock = ManualClock()
    bucket = TokenBucket(rate_per_second=rate, capacity=capacity, clock=clock)

    for delta, amount in ops:
        clock.advance(delta)
        if amount > capacity:
            _assert_amount_above_capacity_raises(bucket, amount)
            continue
        allowed, retry_after = bucket.try_acquire(amount)
        if allowed:
            assert retry_after == pytest.approx(0.0)
            continue
        assert retry_after > 0.0
        clock.advance(retry_after + _RETRY_SLACK)
        retried, second_retry_after = bucket.try_acquire(amount)
        assert retried
        assert second_retry_after == pytest.approx(0.0)


@given(scenario=_bucket_scenarios())
def test_token_bucket_grants_stay_within_capacity_plus_refill(
    scenario: _BucketScenario,
) -> None:
    rate, capacity, ops = scenario
    clock = ManualClock()
    bucket = TokenBucket(rate_per_second=rate, capacity=capacity, clock=clock)
    granted = 0.0
    elapsed = 0.0

    for delta, amount in ops:
        clock.advance(delta)
        elapsed += delta
        if amount > capacity:
            _assert_amount_above_capacity_raises(bucket, amount)
        elif bucket.try_acquire(amount)[0]:
            granted += amount
        budget = capacity + elapsed * rate
        assert granted <= budget * (1.0 + 1e-9) + 1e-9


@given(scenario=_bucket_scenarios())
def test_token_bucket_retry_after_at_most_full_refill_wait(
    scenario: _BucketScenario,
) -> None:
    rate, capacity, ops = scenario
    clock = ManualClock()
    bucket = TokenBucket(rate_per_second=rate, capacity=capacity, clock=clock)

    for delta, amount in ops:
        clock.advance(delta)
        if amount > capacity:
            _assert_amount_above_capacity_raises(bucket, amount)
            continue
        allowed, retry_after = bucket.try_acquire(amount)
        if allowed:
            assert retry_after == pytest.approx(0.0)
            continue
        assert retry_after > 0.0
        assert retry_after <= amount / rate * (1.0 + 1e-9) + 1e-9


@given(rate=_RATES, burst=_BURSTS, deltas=st.lists(_DELTAS, max_size=20))
def test_token_bucket_throttler_denied_check_succeeds_after_retry_after(
    rate: float,
    burst: float,
    deltas: list[float],
) -> None:
    clock = ManualClock()
    throttler = TokenBucketThrottler(
        rate_per_second=rate,
        key=lambda request: "global",
        burst=burst,
        clock=clock,
    )
    request = _recorded("GET", "/")

    for delta in deltas:
        clock.advance(delta)
        decision = throttler.check(request)
        if decision.allowed:
            assert decision.retry_after_seconds == pytest.approx(0.0)
            continue
        assert decision.retry_after_seconds > 0.0
        clock.advance(decision.retry_after_seconds + _RETRY_SLACK)
        assert throttler.check(request).allowed
