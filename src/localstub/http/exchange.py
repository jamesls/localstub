from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from localstub.http.request import RecordedHTTPRequest
from localstub.http.response import RecordedHTTPResponse


@dataclass(frozen=True)
class RecordedExchange:
    request: RecordedHTTPRequest
    response: RecordedHTTPResponse | None
    request_timestamp: datetime
    response_timestamp: datetime | None
