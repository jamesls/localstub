from __future__ import annotations

import httpx

from localstub.http.client import HTTPClientError
from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest
from localstub.http.responsespec import HTTPResponse


async def _read_raw_body(response: httpx.Response) -> tuple[bytes, bool]:
    """Read the response body, returning it and whether it is decoded."""
    try:
        body = bytearray()
        async for chunk in response.aiter_raw():
            body.extend(chunk)
        return bytes(body), False
    except httpx.StreamConsumed:
        # A response hook (or MockTransport built from ``content=``)
        # already consumed the raw stream; httpx caches only the
        # decoded body.
        return response.content, True


class HttpxClient:
    """HTTPClient backed by an injected httpx.AsyncClient."""

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        outgoing = self._client.build_request(
            method=request.method,
            url=request.target,
            headers=list(request.headers.items()),
            content=request.body,
        )
        # httpx's generated framing collapses the distinction between
        # absent content (None) and empty content (b""): it adds
        # Content-Length: 0 to bodyless POST/PUT/PATCH requests and
        # omits it for empty-body requests with other methods.  Framing
        # must follow the source body.
        if request.body is None:
            outgoing.headers.pop("Content-Length", None)
        elif "Content-Length" not in outgoing.headers:
            outgoing.headers["Content-Length"] = str(len(request.body))
        try:
            response = await self._client.send(outgoing, stream=True)
            try:
                body, decoded = await _read_raw_body(response)
            finally:
                await response.aclose()
        except httpx.RequestError as exc:
            raise HTTPClientError(str(exc)) from exc
        items = response.headers.multi_items()
        if decoded:
            items = [
                (name, value)
                for name, value in items
                if name.lower() not in ("content-encoding", "content-length")
            ]
        return HTTPResponse(
            status=response.status_code,
            headers=Headers.from_items(items),
            body=body,
        )
