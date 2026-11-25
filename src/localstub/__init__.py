"""localstub: an asyncio HTTP test server for testing HTTP clients.

The names exported here are localstub's public API.  Deeper modules
(``localstub.http``, ``localstub.middleware``, and friends) are
internal and may change without notice.
"""

from __future__ import annotations

from localstub.http.clients.asyncio import AsyncioClient
from localstub.http.exchange import (
    ClosePhase,
    CloseReason,
    ConnectionClosed,
    RecordedExchange,
)
from localstub.http.headers import Headers
from localstub.http.request import (
    HTTPRequest,
    HTTPRequestHeaders,
    RecordedHTTPRequest,
)
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HTTPResponse, Pause
from localstub.http.uri import ParsedURI
from localstub.middleware import (
    CloseConnection,
    CloseDuringRequest,
    HeaderContext,
    HeaderDecision,
    HeaderMiddleware,
    HeaderNext,
    ResponderContext,
    ResponderMiddleware,
    ResponderNext,
    ResponseSpec,
    SenderContext,
    SenderMiddleware,
    SenderNext,
    SendResult,
)
from localstub.middleware.builtins import close_during_request
from localstub.server import (
    AbortTransmission,
    AsyncHTTPTestServer,
    ByteFlip,
    Delay,
    DropConnection,
    FaultyTransmission,
    ImmediateTransmission,
    OnHeadersReceived,
    SendResponse,
    ThrottledTransmission,
    TruncateBody,
)
from localstub.tlsproxy import AsyncTLSInterceptProxy
from localstub.traffic_jsonl import (
    JSONLTrafficWriter,
    dump_server_traffic_jsonl,
    exchange_to_json_obj,
    record_traffic_jsonl,
)

__all__ = [
    "AbortTransmission",
    "AsyncHTTPTestServer",
    "AsyncTLSInterceptProxy",
    "AsyncioClient",
    "ByteFlip",
    "CloseConnection",
    "CloseDuringRequest",
    "ClosePhase",
    "CloseReason",
    "ConnectionClosed",
    "Delay",
    "DropConnection",
    "FaultyTransmission",
    "HTTPRequest",
    "HTTPRequestHeaders",
    "HTTPResponse",
    "HeaderContext",
    "HeaderDecision",
    "HeaderMiddleware",
    "HeaderNext",
    "Headers",
    "ImmediateTransmission",
    "JSONLTrafficWriter",
    "OnHeadersReceived",
    "ParsedURI",
    "Pause",
    "RecordedExchange",
    "RecordedHTTPRequest",
    "RecordedHTTPResponse",
    "ResponderContext",
    "ResponderMiddleware",
    "ResponderNext",
    "ResponseSpec",
    "SendResponse",
    "SendResult",
    "SenderContext",
    "SenderMiddleware",
    "SenderNext",
    "ThrottledTransmission",
    "TruncateBody",
    "close_during_request",
    "dump_server_traffic_jsonl",
    "exchange_to_json_obj",
    "record_traffic_jsonl",
]
