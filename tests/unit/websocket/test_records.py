from __future__ import annotations

from datetime import UTC, datetime

import pytest

from localstub.http.exchange import ConnectionClosed, RecordedExchange
from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.websocket.deflate import PerMessageDeflate
from localstub.websocket.frames import WebSocketFrame
from localstub.websocket.records import (
    BoundedHistory,
    RecordedWebSocket,
    RecordedWebSocketFrame,
    RecordedWebSocketMessage,
    WebSocketClose,
    WebSocketLog,
    WebSocketMessage,
)

TIMESTAMP = datetime(2026, 11, 24, tzinfo=UTC)
REQUEST = HTTPRequest(method="GET", target="/")
HANDSHAKE = RecordedExchange(
    request=RecordedHTTPRequest(
        request=REQUEST,
        as_received=REQUEST,
        wire_raw_bytes=b"",
        http_version="1.1",
    ),
    response=None,
    request_timestamp=TIMESTAMP,
    response_timestamp=None,
)
CLOSED = ConnectionClosed(
    client=None,
    reason="websocket_close",
    phase="websocket",
    reset=False,
    requests_completed=1,
    bytes_read=0,
    bytes_consumed=0,
    bytes_written=0,
    timestamp=TIMESTAMP,
)


def _log() -> WebSocketLog:
    return WebSocketLog(
        handshake=HANDSHAKE,
        subprotocol="rpc.v1",
        compression=PerMessageDeflate(),
        history_size=None,
    )


def test_recorded_message_delegates_to_its_message() -> None:
    frame = RecordedWebSocketFrame(
        frame=WebSocketFrame.text("hé"),
        sender="client",
        wire_raw_bytes=b"",
        timestamp=TIMESTAMP,
        monotonic=0.0,
    )
    message = RecordedWebSocketMessage(
        message=WebSocketMessage("text", "hé".encode()),
        sender="client",
        frames=(frame,),
        compressed=False,
        timestamp=TIMESTAMP,
    )

    assert message.kind == "text"
    assert message.data == "hé".encode()
    assert message.text == "hé"


def test_message_text_replaces_invalid_utf8() -> None:
    assert WebSocketMessage("text", b"a\xff").text == "a\ufffd"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"", WebSocketClose(None)),
        (b"\x03", WebSocketClose(None)),
        (b"\x03\xe8", WebSocketClose(1000)),
        (b"\x0f\xa0b\xffy", WebSocketClose(4000, "b\ufffdy")),
    ],
)
def test_close_from_payload(payload: bytes, expected: WebSocketClose) -> None:
    assert WebSocketClose.from_payload(payload) == expected


def test_first_close_per_sender_wins() -> None:
    log = _log()
    recorded = RecordedWebSocket(log)

    log.set_close("client", WebSocketClose(1000))
    log.set_close("client", WebSocketClose(1001))
    log.set_close("server", WebSocketClose(4000))
    log.set_close("server", WebSocketClose(4001))

    assert recorded.client_close == WebSocketClose(1000)
    assert recorded.server_close == WebSocketClose(4000)


@pytest.mark.asyncio
async def test_end_is_idempotent_and_wakes_waiters() -> None:
    log = _log()
    recorded = RecordedWebSocket(log)

    log.end(CLOSED)
    log.end(None)
    await recorded.wait_ended(timeout=1.0)

    assert recorded.ended
    assert recorded.closed is CLOSED


def test_view_exposes_handshake_agreement() -> None:
    recorded = RecordedWebSocket(_log())

    assert recorded.handshake is HANDSHAKE
    assert recorded.subprotocol == "rpc.v1"
    assert recorded.compression == PerMessageDeflate()
    assert recorded.messages == []
    assert repr(recorded) == (
        "RecordedWebSocket(frames=0, messages=0, violations=0, ended=False)"
    )


def test_bounded_history_evicts_oldest_past_byte_bound() -> None:
    history: BoundedHistory[bytes] = BoundedHistory(None, 10, len)

    for item in (b"aaaa", b"bbbb", b"cccc"):
        history.append(item)

    assert list(history) == [b"bbbb", b"cccc"]
    assert history.dropped == 1


def test_bounded_history_keeps_newest_entry_larger_than_bound() -> None:
    history: BoundedHistory[bytes] = BoundedHistory(5, 10, len)
    history.append(b"small")

    history.append(b"x" * 50)

    assert list(history) == [b"x" * 50]
    assert history.dropped == 1


def test_bounded_history_popleft_frees_its_bytes() -> None:
    history: BoundedHistory[bytes] = BoundedHistory(None, 8, len)
    history.append(b"aaaa")
    history.append(b"bbbb")

    assert history.popleft() == b"aaaa"
    history.append(b"cccc")

    assert list(history) == [b"bbbb", b"cccc"]
    assert history.dropped == 0


def test_log_bounds_frames_and_messages_by_bytes() -> None:
    log = WebSocketLog(
        handshake=HANDSHAKE,
        subprotocol=None,
        compression=None,
        history_size=None,
        history_bytes=100,
    )
    recorded = RecordedWebSocket(log)
    for index in range(5):
        frame = RecordedWebSocketFrame(
            frame=WebSocketFrame.binary(bytes(20)),
            sender="client",
            wire_raw_bytes=bytes(22),
            timestamp=TIMESTAMP,
            monotonic=float(index),
        )
        log.add_frame(frame)
        log.add_message(
            RecordedWebSocketMessage(
                message=WebSocketMessage("binary", bytes(20)),
                sender="client",
                frames=(frame,),
                compressed=False,
                timestamp=TIMESTAMP,
            )
        )

    # A frame record holds 42 bytes and a message record 62.
    assert [f.monotonic for f in recorded.frames] == [3.0, 4.0]
    assert recorded.dropped_frames == 3
    assert len(recorded.messages) == 1
    assert recorded.dropped_messages == 4
