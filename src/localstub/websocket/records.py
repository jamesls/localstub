"""What a WebSocket session leaves behind: frames, messages, closes.

The frame and message records follow the split localstub uses for
HTTP: a value (``WebSocketFrame``, ``WebSocketMessage``) plus the
evidence of how it crossed the wire.  Records name a ``sender``
rather than a direction relative to localstub, so the same types
describe a stub session and a relayed one.

``RecordedWebSocket`` is the read-only view a test reads.  The session
and the relay append through the ``WebSocketLog`` it wraps.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from localstub.http.exchange import ConnectionClosed, RecordedExchange
from localstub.websocket.deflate import PerMessageDeflate
from localstub.websocket.frames import WebSocketFrame

type Sender = Literal["client", "server"]
type MessageKind = Literal["text", "binary"]
type ViolationKind = Literal[
    "unmasked_frame",
    "masked_frame",
    "reserved_bits",
    "reserved_opcode",
    "fragmented_control_frame",
    "oversized_control_frame",
    "unexpected_continuation",
    "interrupted_message",
    "invalid_length",
    "invalid_close",
    "frame_after_close",
    "invalid_utf8",
    "invalid_compressed_data",
    "message_too_big",
]


@dataclass(frozen=True)
class RecordedWebSocketFrame:
    """A frame as it crossed the wire.

    ``frame.payload`` is the unmasked payload and, for a compressed
    message, still compressed.  ``wire_raw_bytes`` are the exact bytes
    that carried the frame.  ``monotonic`` comes from the injected
    ``Clock``, for interval assertions such as ping spacing.
    """

    frame: WebSocketFrame
    sender: Sender
    wire_raw_bytes: bytes
    timestamp: datetime
    monotonic: float


@dataclass(frozen=True)
class WebSocketMessage:
    """One complete data message: its kind and its application bytes."""

    kind: MessageKind
    data: bytes

    @property
    def text(self) -> str:
        """The data decoded as UTF-8, with replacement."""
        return self.data.decode("utf-8", errors="replace")


@dataclass(frozen=True)
class RecordedWebSocketMessage:
    """A message plus the data frames that carried it.

    ``data`` is the inflated message when ``compressed`` is ``True``;
    the frames keep the compressed payloads.
    """

    message: WebSocketMessage
    sender: Sender
    frames: tuple[RecordedWebSocketFrame, ...]
    compressed: bool
    timestamp: datetime

    @property
    def kind(self) -> MessageKind:
        return self.message.kind

    @property
    def data(self) -> bytes:
        return self.message.data

    @property
    def text(self) -> str:
        return self.message.text


@dataclass(frozen=True)
class WebSocketClose:
    """The code and reason of a Close frame.

    ``code`` is ``None`` for a Close frame with an empty payload.
    """

    code: int | None
    reason: str = ""

    @classmethod
    def from_payload(cls, payload: bytes) -> WebSocketClose:
        """Read a Close payload, decoding the reason with replacement.

        A payload shorter than two bytes carries no code.
        """
        if len(payload) < 2:
            return cls(None)
        reason = payload[2:].decode("utf-8", errors="replace")
        return cls(int.from_bytes(payload[:2], "big"), reason)


@dataclass(frozen=True)
class WebSocketViolation:
    """An RFC 6455 or RFC 7692 rule a peer broke.

    ``frame`` is the frame that broke it, or ``None`` when the frame
    was refused from its header before its payload arrived.
    """

    kind: ViolationKind
    close_code: int
    sender: Sender
    description: str
    frame: RecordedWebSocketFrame | None = None


def frame_size(frame: RecordedWebSocketFrame) -> int:
    """Bytes a frame record holds: its wire bytes and its payload."""
    return len(frame.wire_raw_bytes) + len(frame.frame.payload)


def message_size(message: RecordedWebSocketMessage) -> int:
    """Bytes a message record holds, including the frames it keeps."""
    return len(message.data) + sum(frame_size(f) for f in message.frames)


def _violation_size(violation: WebSocketViolation) -> int:
    frame = violation.frame
    return 0 if frame is None else frame_size(frame)


class BoundedHistory[T]:
    """Oldest-first history bounded by entry count and by bytes.

    Appending evicts the oldest entries until both bounds hold, and
    counts them in ``dropped``.  The newest entry is always kept, so a
    single entry larger than ``max_bytes`` is still recorded.  ``None``
    leaves a bound off.
    """

    def __init__(
        self,
        maxlen: int | None,
        max_bytes: int | None,
        size: Callable[[T], int],
    ) -> None:
        self._items: deque[T] = deque()
        self._maxlen = maxlen
        self._max_bytes = max_bytes
        self._size = size
        self._bytes = 0
        self.dropped = 0

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[T]:
        return iter(self._items)

    def append(self, item: T) -> None:
        self._items.append(item)
        self._bytes += self._size(item)
        while len(self._items) > 1 and self._over():
            self._bytes -= self._size(self._items.popleft())
            self.dropped += 1

    def popleft(self) -> T:
        item = self._items.popleft()
        self._bytes -= self._size(item)
        return item

    def _over(self) -> bool:
        if self._maxlen is not None and len(self._items) > self._maxlen:
            return True
        return self._max_bytes is not None and self._bytes > self._max_bytes


class WebSocketLog:
    """The mutable record behind a ``RecordedWebSocket``.

    The frame, message and violation histories each hold at most
    ``history_size`` entries and ``history_bytes`` bytes, evicting the
    oldest and counting what they evict; ``None`` leaves a bound off.
    """

    def __init__(
        self,
        *,
        handshake: RecordedExchange,
        subprotocol: str | None,
        compression: PerMessageDeflate | None,
        history_size: int | None,
        history_bytes: int | None = None,
    ) -> None:
        self.handshake = handshake
        self.subprotocol = subprotocol
        self.compression = compression
        self.frames: BoundedHistory[RecordedWebSocketFrame] = BoundedHistory(
            history_size, history_bytes, frame_size
        )
        self.messages: BoundedHistory[RecordedWebSocketMessage] = (
            BoundedHistory(history_size, history_bytes, message_size)
        )
        self.violations: BoundedHistory[WebSocketViolation] = BoundedHistory(
            history_size, history_bytes, _violation_size
        )
        self.client_close: WebSocketClose | None = None
        self.server_close: WebSocketClose | None = None
        self.closed: ConnectionClosed | None = None
        self.ended = asyncio.Event()

    def add_frame(self, frame: RecordedWebSocketFrame) -> None:
        self.frames.append(frame)

    def add_message(self, message: RecordedWebSocketMessage) -> None:
        self.messages.append(message)

    def add_violation(self, violation: WebSocketViolation) -> None:
        self.violations.append(violation)

    def set_close(self, sender: Sender, close: WebSocketClose) -> None:
        """Record the first Close frame ``sender`` sent."""
        if sender == "client":
            if self.client_close is None:
                self.client_close = close
        elif self.server_close is None:
            self.server_close = close

    def end(self, closed: ConnectionClosed | None) -> None:
        """Mark the session ended, with the connection's close event."""
        if self.ended.is_set():
            return
        self.closed = closed
        self.ended.set()


class RecordedWebSocket:
    """Everything recorded about one WebSocket session.

    ``frames`` and ``messages`` hold both directions in the order
    localstub read or wrote them; the ``client_*`` and ``server_*``
    members filter them by sender.  ``closed`` is the connection's
    ``ConnectionClosed`` once the session ends; it stays ``None`` for
    a session relayed in forward mode, which publishes no close
    events.
    """

    def __init__(self, log: WebSocketLog) -> None:
        self._log = log

    def __repr__(self) -> str:
        log = self._log
        return (
            f"RecordedWebSocket(frames={len(log.frames)}, "
            f"messages={len(log.messages)}, "
            f"violations={len(log.violations)}, "
            f"ended={log.ended.is_set()})"
        )

    @property
    def handshake(self) -> RecordedExchange:
        """The exchange of the upgrade request and the 101."""
        return self._log.handshake

    @property
    def subprotocol(self) -> str | None:
        return self._log.subprotocol

    @property
    def compression(self) -> PerMessageDeflate | None:
        """The agreed permessage-deflate parameters, if negotiated."""
        return self._log.compression

    @property
    def frames(self) -> list[RecordedWebSocketFrame]:
        return list(self._log.frames)

    @property
    def messages(self) -> list[RecordedWebSocketMessage]:
        return list(self._log.messages)

    @property
    def client_frames(self) -> list[RecordedWebSocketFrame]:
        return [f for f in self._log.frames if f.sender == "client"]

    @property
    def server_frames(self) -> list[RecordedWebSocketFrame]:
        return [f for f in self._log.frames if f.sender == "server"]

    @property
    def client_messages(self) -> list[RecordedWebSocketMessage]:
        return [m for m in self._log.messages if m.sender == "client"]

    @property
    def server_messages(self) -> list[RecordedWebSocketMessage]:
        return [m for m in self._log.messages if m.sender == "server"]

    @property
    def client_close(self) -> WebSocketClose | None:
        return self._log.client_close

    @property
    def server_close(self) -> WebSocketClose | None:
        return self._log.server_close

    @property
    def violations(self) -> list[WebSocketViolation]:
        return list(self._log.violations)

    @property
    def dropped_frames(self) -> int:
        """Frames evicted from the bounded ``frames`` history."""
        return self._log.frames.dropped

    @property
    def dropped_messages(self) -> int:
        """Messages evicted from the bounded ``messages`` history."""
        return self._log.messages.dropped

    @property
    def ended(self) -> bool:
        return self._log.ended.is_set()

    @property
    def closed(self) -> ConnectionClosed | None:
        return self._log.closed

    async def wait_ended(self, timeout: float | None = None) -> None:
        """Wait for the session to end."""
        await asyncio.wait_for(self._log.ended.wait(), timeout)
