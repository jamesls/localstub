from __future__ import annotations

import asyncio
import logging

from localstub.forward import open_upstream_connection
from localstub.http.client import HTTPClientError
from localstub.http.request import HTTPRequest
from localstub.http.response import AsyncMultiResponseParser
from localstub.http.responsespec import HTTPResponse
from localstub.http.uri import ParsedURI
from localstub.http.utils import headers_to_headers

LOG = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT = 30.0
DEFAULT_READ_TIMEOUT = 300.0


def _serialize_request(request: HTTPRequest, uri: ParsedURI) -> bytes:
    """Serialize *request* to origin-form HTTP/1.1 wire bytes.

    Host and Content-Length are generated here; the caller guarantees
    the request headers are already free of them (adapter contract).
    """
    lines = [
        f"{request.method} {uri.path or '/'} HTTP/1.1",
        f"Host: {uri.authority}",
    ]
    for name, value in request.headers.items():
        lines.append(f"{name}: {value}")
    if request.body is not None:
        lines.append(f"Content-Length: {len(request.body)}")
    lines.append("Connection: close")
    head = "\r\n".join(lines).encode("latin-1") + b"\r\n\r\n"
    return head + (request.body or b"")


class AsyncioClient:
    """Built-in HTTPClient backed by asyncio streams.

    Uses machinery this package already owns: asyncio stream
    connections with truststore-verified TLS and the httptools-backed
    response parser.  No third-party HTTP client library.

    One TCP (+TLS) handshake per ``send()``; no connection pooling,
    redirects, retries, cookies, or content decoding.  Inject
    ``HttpxClient`` instead when forwarding heavy traffic that needs
    pooling or HTTP/2.
    """

    def __init__(
        self,
        *,
        connect_timeout: float | None = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float | None = DEFAULT_READ_TIMEOUT,
        verify_tls: bool = True,
        max_read: int = 8192,
    ) -> None:
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._verify_tls = verify_tls
        self._max_read = max_read

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        uri = request.target_uri
        if uri is None:
            raise HTTPClientError(
                f"request target is not absolute-form: {request.target!r}"
            )
        wire = _serialize_request(request, uri)
        try:
            async with asyncio.timeout(self._connect_timeout):
                reader, writer = await open_upstream_connection(
                    uri.host,
                    uri.port,
                    use_tls=uri.scheme == "https",
                    verify=self._verify_tls,
                )
        except (TimeoutError, OSError) as exc:
            raise HTTPClientError(
                f"failed to connect to {uri.host}:{uri.port}: {exc}"
            ) from exc

        try:
            async with asyncio.timeout(self._read_timeout):
                writer.write(wire)
                await writer.drain()
                return await self._read_final_response(reader, request.method)
        except (TimeoutError, OSError) as exc:
            raise HTTPClientError(
                f"exchange with {uri.host}:{uri.port} failed: {exc}"
            ) from exc
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:  # pragma: no cover - close/reset race
                LOG.debug("Failed to close upstream writer", exc_info=True)

    async def _read_final_response(
        self,
        reader: asyncio.StreamReader,
        request_method: str,
    ) -> HTTPResponse:
        parser = AsyncMultiResponseParser(max_read=self._max_read)
        while True:
            parsed, _ = await parser.next_response(reader, request_method)
            if parsed is None:
                raise HTTPClientError("failed to parse upstream response")
            status = parsed.status_code or 0
            if status >= 200:
                return HTTPResponse(
                    status=status,
                    headers=headers_to_headers(parsed.headers),
                    body=parsed.body,
                )
            LOG.debug(
                "Skipping 1xx informational response (%d) while "
                "waiting for the final response",
                status,
            )
