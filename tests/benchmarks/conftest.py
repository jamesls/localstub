from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest


@pytest.fixture
def loop() -> Iterator[asyncio.AbstractEventLoop]:
    # The benchmark fixture measures a synchronous callable, so each
    # benchmark drives its coroutine with run_until_complete() on a loop
    # created here, outside the measured region.  asyncio.run() inside
    # the callable would charge loop setup and teardown to every sample.
    event_loop = asyncio.new_event_loop()
    try:
        yield event_loop
    finally:
        event_loop.close()
