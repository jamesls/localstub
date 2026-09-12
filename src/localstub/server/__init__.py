"""The asyncio HTTP test server.

``core`` holds ``AsyncHTTPTestServer``, ``connection`` the per-connection
loop and state, and ``transmission`` the response body strategies and
fault steps.  Every name the former ``localstub.server`` module exported
is re-exported here, so public imports do not change.
"""

from __future__ import annotations

from localstub.http.exchange import (
    ClosePhase,
    CloseReason,
    ConnectionClosed,
    RecordedExchange,
)
from localstub.http.request import (
    HTTPRequest,
    HTTPRequestHeaders,
    RecordedHTTPRequest,
)
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HTTPResponse
from localstub.middleware import (
    CloseConnection,
    CloseDuringRequest,
    HeaderDecision,
    ResponseSpec,
    SendResult,
)
from localstub.router import ResponderHandler
from localstub.server.connection import (
    ConnectionState,
    CountingStreamReader,
    HTTPConnection,
    KeepAlivePolicy,
    RecordingStreamWriter,
    RequestPipeline,
    pack_linger_option,
)
from localstub.server.core import (
    AsyncHTTPTestServer,
    OnHeadersReceived,
    SendResponse,
    ThrottleResponse,
)
from localstub.server.transmission import (
    AbortTransmission,
    ApplyResult,
    ByteFlip,
    Delay,
    DropConnection,
    FaultStep,
    FaultyTransmission,
    ImmediateTransmission,
    ThrottledTransmission,
    TransmissionStrategy,
    TruncateBody,
    Writer,
)

__all__ = [
    "AbortTransmission",
    "ApplyResult",
    "AsyncHTTPTestServer",
    "ByteFlip",
    "CloseConnection",
    "CloseDuringRequest",
    "ClosePhase",
    "CloseReason",
    "ConnectionClosed",
    "ConnectionState",
    "CountingStreamReader",
    "Delay",
    "DropConnection",
    "FaultStep",
    "FaultyTransmission",
    "HTTPConnection",
    "HTTPRequest",
    "HTTPRequestHeaders",
    "HTTPResponse",
    "HeaderDecision",
    "ImmediateTransmission",
    "KeepAlivePolicy",
    "OnHeadersReceived",
    "RecordedExchange",
    "RecordedHTTPRequest",
    "RecordedHTTPResponse",
    "RecordingStreamWriter",
    "RequestPipeline",
    "ResponderHandler",
    "ResponseSpec",
    "SendResponse",
    "SendResult",
    "ThrottleResponse",
    "ThrottledTransmission",
    "TransmissionStrategy",
    "TruncateBody",
    "Writer",
    "pack_linger_option",
]
