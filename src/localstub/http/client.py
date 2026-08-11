"""localstub's definition of an HTTP client.

Adapters over concrete backends live in ``localstub.http.clients``.
"""

from __future__ import annotations

from typing import Protocol

from localstub.http.request import HTTPRequest
from localstub.http.responsespec import HTTPResponse


class HTTPClient(Protocol):
    """localstub's definition of an HTTP client: one exchange per send().

    Normative requirements on any implementation:

    1. Exactly one HTTP request per ``send()``.  The adapter itself
       never follows redirects, retries, or reads proxy environment
       variables.  Configuration the user explicitly set on an injected
       backend client is theirs.
    2. Host and body framing (Content-Length) are derived from
       ``HTTPRequest`` fields.  The caller guarantees ``headers`` is
       already free of hop-by-hop headers, Content-Length, and Host.
    3. The headers returned must describe the body returned.  Body comes
       back as received on the wire when the backend permits; if the
       backend can only produce a content-decoded body, the adapter
       strips Content-Encoding/Content-Length itself before returning.
    4. Response header order and duplicates are preserved.
    5. Connect/TLS/timeout/protocol failures raise ``HTTPClientError``
       (cause chained).  Anything else propagating is an adapter bug.
    6. Upstream TLS is verified by default using system trust.
    7. The adapter never closes an injected backend client.  Lifecycle
       belongs to whoever constructed it.
    """

    async def send(self, request: HTTPRequest) -> HTTPResponse: ...


class HTTPClientError(Exception):
    """The exchange could not be completed (connect/TLS/timeout/protocol)."""
