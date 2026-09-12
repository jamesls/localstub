from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from localstub.http.request import RecordedHTTPRequest
from localstub.http.response import RecordedHTTPResponse

type CloseReason = Literal[
    "close_response",  # CloseConnection from a responder
    "request_read",  # CloseDuringRequest or header middleware False
    "response_aborted",  # DropConnection cut the response body
    "idle_timeout",  # keep_alive_timeout expired, or is 0.0
    "max_requests",  # max_requests_per_connection reached
    "connection_close",  # normal HTTP connection-persistence rules
    "client",  # client closed first (EOF or reset observed)
    "protocol_error",  # the request could not be parsed
    "error",  # handler or middleware raised
    "shutdown",  # server or proxy shutdown closed it
]

type ClosePhase = Literal[
    "idle",  # no bytes of the next request yet
    "request_headers",  # request line or headers incomplete
    "request_body",  # headers parsed, body short of its boundary
    "response",  # request complete, no responder output written
    "response_body",  # response head written, body unfinished
    "after_response",  # response fully written
]


@dataclass(frozen=True)
class ConnectionClosed:
    """The one terminal event recorded for every HTTP connection.

    ``reason`` names the cause and ``phase`` names where the connection
    was in the request lifecycle when the close was decided.  For a
    server-initiated close, ``reset`` records the requested close mode;
    for reason ``client`` it is ``True`` only when the server observed
    a reset.  It never reports what the remote client saw.

    ``requests_completed`` counts requests read to their message
    boundary on this connection, whether or not a response was sent.
    The byte counters are snapshotted when the close is decided:
    ``bytes_read`` is what the server read from the stream,
    ``bytes_consumed`` what the parser attributed to requests, and
    ``bytes_written`` what was passed to the writer.
    """

    client: tuple[str, int] | None
    reason: CloseReason
    phase: ClosePhase
    reset: bool
    requests_completed: int
    bytes_read: int
    bytes_consumed: int
    bytes_written: int
    timestamp: datetime


@dataclass(frozen=True)
class RecordedExchange:
    """A recorded request with the response and close event it owns.

    ``interim_responses`` holds the interim (1xx other than 101)
    responses sent during the header phase.  ``closed`` is set when the
    connection's close was decided while this exchange was in progress;
    it is the same object that appears in the closed-connection history.
    """

    request: RecordedHTTPRequest
    response: RecordedHTTPResponse | None
    request_timestamp: datetime
    response_timestamp: datetime | None
    interim_responses: tuple[RecordedHTTPResponse, ...] = ()
    closed: ConnectionClosed | None = None
