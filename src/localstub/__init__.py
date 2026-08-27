"""localstub: an asyncio HTTP test server for testing HTTP clients.

The names exported here are localstub's public API.  Deeper modules
(``localstub.http``, ``localstub.middleware``, and friends) are
internal and may change without notice.
"""

from __future__ import annotations

from localstub.http.clients.asyncio import AsyncioClient
from localstub.http.exchange import RecordedExchange
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
    HeaderContext,
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
from localstub.server import (
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

__all__ = [
    "AsyncHTTPTestServer",
    "AsyncTLSInterceptProxy",
    "AsyncioClient",
    "ByteFlip",
    "Delay",
    "DropConnection",
    "FaultyTransmission",
    "HTTPRequest",
    "HTTPRequestHeaders",
    "HTTPResponse",
    "HeaderContext",
    "HeaderMiddleware",
    "HeaderNext",
    "Headers",
    "ImmediateTransmission",
    "OnHeadersReceived",
    "ParsedURI",
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
    "dump_server_traffic_jsonl",
]
