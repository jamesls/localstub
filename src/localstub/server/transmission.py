"""Response body transmission strategies and fault steps."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


class Writer(Protocol):
    """What a transmission strategy may do with the connection.

    Strategies write and drain.  Closing is the connection loop's job:
    a strategy that stops early returns ``AbortTransmission`` and the
    loop decides and performs the close, so every close is recorded.
    """

    def write(self, data: bytes) -> None: ...

    async def drain(self) -> None: ...


@dataclass(frozen=True)
class AbortTransmission:
    """Returned by a strategy that stopped writing and wants the
    connection closed."""

    reset: bool = False
    """Request an abortive TCP close instead of ordinary closure."""


class TransmissionStrategy:
    """Protocol for controlling how response body bytes are transmitted.

    This allows tests to simulate network conditions like slow transfers,
    throttled bandwidth, etc. without changing the actual response content.
    """

    async def write_body(
        self,
        writer: Writer,
        body: bytes,
    ) -> AbortTransmission | None:
        """Write the response body to the client.

        Args:
            writer: The stream writer to write to
            body: The complete response body bytes to transmit

        Returns:
            ``None`` when the whole body was written, or an
            ``AbortTransmission`` when the strategy stopped early and
            the connection must close.
        """
        raise NotImplementedError


class ImmediateTransmission(TransmissionStrategy):
    """Default transmission strategy - send entire body immediately."""

    async def write_body(
        self,
        writer: Writer,
        body: bytes,
    ) -> AbortTransmission | None:
        writer.write(body)
        await writer.drain()
        return None


class ThrottledTransmission(TransmissionStrategy):
    """Throttled transmission strategy - send body in chunks with delays.

    Useful for testing client behavior with slow network connections or
    bandwidth-limited scenarios (e.g., S3 GetObject with slow transfer).
    """

    def __init__(
        self,
        chunk_size: int,
        delay: float,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        """Initialize throttled transmission.

        Args:
            chunk_size: Number of bytes to send in each chunk
            delay: Seconds to wait between chunks
            sleep: Coroutine function used to wait between chunks,
                defaults to ``asyncio.sleep``
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be a positive integer")
        self.chunk_size = chunk_size
        self.delay = delay
        self._sleep = sleep

    async def write_body(
        self,
        writer: Writer,
        body: bytes,
    ) -> AbortTransmission | None:
        offset = 0
        while offset < len(body):
            chunk = body[offset : offset + self.chunk_size]
            writer.write(chunk)
            await writer.drain()

            offset += self.chunk_size
            if offset < len(body):  # Don't delay after last chunk
                await self._sleep(self.delay)
        return None


@dataclass
class ApplyResult:
    """Result of applying a fault step."""

    body: bytes
    delay_before: float = 0.0
    drop_after: int | None = None
    drop_reset: bool = False


class FaultStep(Protocol):
    """Protocol for fault steps that mutate transmission behavior."""

    def apply(self, body: bytes) -> ApplyResult: ...


class Delay(FaultStep):
    """Delay sending the body."""

    def __init__(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("seconds must be non-negative")
        self.seconds = seconds

    def apply(self, body: bytes) -> ApplyResult:
        return ApplyResult(body=body, delay_before=self.seconds)


class DropConnection(FaultStep):
    """Close the connection after sending part of the body.

    The connection loop records the close with reason
    ``response_aborted``; ``reset`` requests an abortive TCP close.
    """

    def __init__(self, after_bytes: int, reset: bool = False) -> None:
        if after_bytes < 0:
            raise ValueError("after_bytes must be non-negative")
        self.after_bytes = after_bytes
        self.reset = reset

    def apply(self, body: bytes) -> ApplyResult:
        return ApplyResult(
            body=body,
            drop_after=self.after_bytes,
            drop_reset=self.reset,
        )


class TruncateBody(FaultStep):
    """Send only the first N bytes of the body."""

    def __init__(self, keep_bytes: int) -> None:
        if keep_bytes < 0:
            raise ValueError("keep_bytes must be non-negative")
        self.keep_bytes = keep_bytes

    def apply(self, body: bytes) -> ApplyResult:
        return ApplyResult(body=body[: self.keep_bytes])


class ByteFlip(FaultStep):
    """Flip a single byte in the body using XOR."""

    def __init__(self, offset: int, mask: int = 0xFF) -> None:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if mask < 0 or mask > 0xFF:
            raise ValueError("mask must be between 0 and 255")
        self.offset = offset
        self.mask = mask

    def apply(self, body: bytes) -> ApplyResult:
        if self.offset >= len(body):
            return ApplyResult(body=body)
        mutated = bytearray(body)
        mutated[self.offset] ^= self.mask
        return ApplyResult(body=bytes(mutated))


class FaultyTransmission(TransmissionStrategy):
    """Always-on fault injection applied during body transmission."""

    def __init__(
        self,
        faults: list[FaultStep],
        base: TransmissionStrategy | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._faults = faults
        self._base = base or ImmediateTransmission()
        self._sleep = sleep

    async def write_body(
        self,
        writer: Writer,
        body: bytes,
    ) -> AbortTransmission | None:
        body_to_send = body
        total_delay = 0.0
        drop_after: int | None = None
        drop_reset = False

        for fault in self._faults:
            result = fault.apply(body_to_send)
            body_to_send = result.body
            total_delay += result.delay_before
            if drop_after is None and result.drop_after is not None:
                drop_after = result.drop_after
                drop_reset = result.drop_reset

        if total_delay > 0:
            await self._sleep(total_delay)

        if drop_after is None:
            return await self._base.write_body(writer, body_to_send)

        to_send = body_to_send[:drop_after]
        if to_send:
            writer.write(to_send)
            await writer.drain()
        return AbortTransmission(reset=drop_reset)
