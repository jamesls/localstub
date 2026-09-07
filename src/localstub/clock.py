from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """Source of monotonic time, injectable for deterministic tests."""

    def now(self) -> float: ...


class MonotonicClock:
    """Default clock backed by ``time.monotonic``."""

    def now(self) -> float:
        return time.monotonic()
