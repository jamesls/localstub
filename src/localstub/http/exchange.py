from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from localstub.http.request import HTTPRequest
from localstub.http.response import RecordedResponse


@dataclass(frozen=True)
class RecordedExchange:
    request: HTTPRequest
    response: RecordedResponse | None
    request_timestamp: datetime
    response_timestamp: datetime | None
