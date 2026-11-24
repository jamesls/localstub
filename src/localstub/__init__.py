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
from localstub.http.responsespec import HTTPResponse
from localstub.http.uri import ParsedURI
from localstub.middleware import (
    AcceptWebSocket,
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
from localstub.traffic_jsonl import dump_server_traffic_jsonl
from localstub.websocket.deflate import PerMessageDeflate
from localstub.websocket.frames import Opcode, WebSocketFrame
from localstub.websocket.records import (
    RecordedWebSocket,
    RecordedWebSocketFrame,
    RecordedWebSocketMessage,
    WebSocketClose,
    WebSocketMessage,
    WebSocketViolation,
)
from localstub.websocket.session import (
    WebSocketEnded,
    WebSocketHandler,
    WebSocketSession,
)

__all__ = [
    "AbortTransmission",
    "AcceptWebSocket",
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
    "OnHeadersReceived",
    "Opcode",
    "ParsedURI",
    "PerMessageDeflate",
    "RecordedExchange",
    "RecordedHTTPRequest",
    "RecordedHTTPResponse",
    "RecordedWebSocket",
    "RecordedWebSocketFrame",
    "RecordedWebSocketMessage",
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
    "WebSocketClose",
    "WebSocketEnded",
    "WebSocketFrame",
    "WebSocketHandler",
    "WebSocketMessage",
    "WebSocketSession",
    "WebSocketViolation",
    "close_during_request",
    "dump_server_traffic_jsonl",
]
