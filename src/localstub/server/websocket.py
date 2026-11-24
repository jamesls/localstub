"""Wire a WebSocket accept to an HTTP connection.

``prepare_accept()`` turns an ``AcceptWebSocket`` into the 101 to
write, or into the ordinary rejection a bad handshake gets.
``build_session()`` builds a ``WebSocketSession`` from a connection's
parts once the 101 is on the wire and the exchange is recorded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from localstub.clock import Clock
from localstub.http import stream
from localstub.http.exchange import RecordedExchange
from localstub.http.headers import Headers
from localstub.http.request import RecordedHTTPRequest
from localstub.http.responsespec import HTTPResponse, normalize_headers
from localstub.middleware import AcceptWebSocket
from localstub.websocket.deflate import agreed_extensions, negotiate
from localstub.websocket.handshake import (
    HandshakeRequest,
    accept_headers,
    validate_handshake,
)
from localstub.websocket.records import WebSocketLog
from localstub.websocket.session import (
    ByteSink,
    CloseSession,
    SessionOptions,
    SessionWriter,
    Sleep,
    WebSocketSession,
)


@dataclass(frozen=True)
class PreparedAccept:
    """The 101 for an accepted handshake, and what it agreed."""

    accept: AcceptWebSocket
    response: HTTPResponse


def prepare_accept(
    request: RecordedHTTPRequest,
    accept: AcceptWebSocket,
) -> PreparedAccept | HTTPResponse:
    """Build the 101 for ``accept``, or the rejection for ``request``.

    Validation reads ``request.as_received``: the client checks the
    accept key against the key it sent, so a middleware rewrite of the
    request must not change the answer.
    """
    handshake = validate_handshake(request.as_received, request.http_version)
    if isinstance(handshake, HTTPResponse):
        return handshake
    return PreparedAccept(
        accept=accept,
        response=HTTPResponse(
            status=101, headers=_response_headers(handshake, accept)
        ),
    )


def _response_headers(
    handshake: HandshakeRequest,
    accept: AcceptWebSocket,
) -> Headers:
    extensions = None
    if accept.compression is not None:
        agreed = negotiate(handshake.extensions, accept.compression)
        if agreed is not None:
            extensions = agreed.header_value()
    return accept_headers(
        handshake,
        subprotocol=accept.subprotocol,
        extensions=extensions,
        extra=normalize_headers(accept.headers),
    )


def build_session(
    *,
    prepared: PreparedAccept,
    handshake: RecordedExchange,
    reader: stream.ByteStream,
    writer: SessionWriter,
    consumed: ByteSink | None,
    close: CloseSession,
    sleep: Sleep,
    clock: Clock,
    timestamp: Callable[[], datetime],
    history_size: int | None,
    history_bytes: int | None,
    write_chunk_size: int | None,
) -> WebSocketSession:
    """Build the session that runs on a switched connection.

    The record holds what the 101 told the client, so a header that
    overrode the computed subprotocol or extensions is what counts.
    ``history_size`` and ``history_bytes`` bound both the record's
    histories and the receive queues.
    """
    headers = prepared.response.headers
    accept = prepared.accept
    log = WebSocketLog(
        handshake=handshake,
        subprotocol=headers.get("Sec-WebSocket-Protocol"),
        compression=agreed_extensions(
            headers.get_all("Sec-WebSocket-Extensions", [])
        ).compression,
        history_size=history_size,
        history_bytes=history_bytes,
    )
    return WebSocketSession(
        reader=reader,
        writer=writer,
        log=log,
        close=close,
        handler=accept.handler,
        options=SessionOptions(
            strict=accept.strict,
            auto_pong=accept.auto_pong,
            auto_close=accept.auto_close,
            close_timeout=accept.close_timeout,
            max_message_bytes=accept.max_message_bytes,
            queue_size=history_size,
            queue_bytes=history_bytes,
            write_chunk_size=write_chunk_size,
        ),
        sleep=sleep,
        clock=clock,
        timestamp=timestamp,
        consumed=consumed,
    )
