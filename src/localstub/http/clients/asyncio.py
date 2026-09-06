from __future__ import annotations

import asyncio
import logging
from typing import Self

from localstub.forward import open_upstream_connection
from localstub.http.client import HTTPClientError
from localstub.http.clients.pool import (
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_MAX_CONNECTIONS_PER_ORIGIN,
    DEFAULT_MAX_IDLE_CONNECTIONS,
    ConnectionPool,
    Origin,
    PooledConnection,
)
from localstub.http.connection import response_allows_reuse
from localstub.http.request import HTTPRequest
from localstub.http.response import AsyncMultiResponseParser, ParsedResponse
from localstub.http.responsespec import HTTPResponse
from localstub.http.uri import ParsedURI
from localstub.http.utils import headers_to_headers, serialize_header_line

LOG = logging.getLogger(__name__)

DEFAULT_CONNECT_TIMEOUT = 30.0
DEFAULT_READ_TIMEOUT = 300.0


def _serialize_request(request: HTTPRequest, uri: ParsedURI) -> bytes:
    """Serialize *request* to origin-form HTTP/1.1 wire bytes.

    Host and Content-Length are generated here; the caller guarantees
    the request headers are already free of them (adapter contract).
    No Connection header is sent: HTTP/1.1 persistence is the default.
    """
    head = bytearray(
        f"{request.method} {uri.path or '/'} HTTP/1.1\r\n".encode("ascii")
    )
    head.extend(serialize_header_line("Host", uri.authority))
    for name, value in request.headers.items():
        head.extend(serialize_header_line(name, value))
    if request.body is not None:
        head.extend(
            serialize_header_line("Content-Length", str(len(request.body)))
        )
    head.extend(b"\r\n")
    return bytes(head) + (request.body or b"")


def _to_http_response(parsed: ParsedResponse) -> HTTPResponse:
    return HTTPResponse(
        status=parsed.status_code or 0,
        headers=headers_to_headers(parsed.headers),
        body=parsed.body,
    )


class AsyncioClient:
    """Built-in HTTPClient backed by asyncio streams.

    Uses machinery this package already owns: asyncio stream
    connections with truststore-verified TLS and the httptools-backed
    response parser.  No third-party HTTP client library.

    Upstream connections are pooled per origin and reused across
    sequential ``send()`` calls; a connection is closed on any error,
    timeout, or ambiguity about its stream position.  No redirects,
    retries, cookies, or content decoding.  ``aclose()`` (or use as an
    async context manager) closes the pool; lifecycle belongs to
    whoever constructed the client.
    """

    def __init__(
        self,
        *,
        connect_timeout: float | None = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float | None = DEFAULT_READ_TIMEOUT,
        verify_tls: bool = True,
        max_read: int = 8192,
        max_connections_per_origin: int = DEFAULT_MAX_CONNECTIONS_PER_ORIGIN,
        max_idle_connections: int = DEFAULT_MAX_IDLE_CONNECTIONS,
        idle_timeout: float | None = DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self._connect_timeout = connect_timeout
        self._read_timeout = read_timeout
        self._verify_tls = verify_tls
        self._max_read = max_read
        self._pool = ConnectionPool(
            self._open_connection,
            max_connections_per_origin=max_connections_per_origin,
            max_idle_connections=max_idle_connections,
            idle_timeout=idle_timeout,
            parser_factory=self._create_parser,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the connection pool and its idle connections."""
        await self._pool.aclose()

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        uri = request.target_uri
        if uri is None:
            raise HTTPClientError(
                f"request target is not absolute-form: {request.target!r}"
            )
        origin = Origin.from_uri(uri)
        wire = _serialize_request(request, uri)
        try:
            async with asyncio.timeout(self._connect_timeout):
                connection = await self._pool.acquire(origin)
        except (TimeoutError, OSError) as exc:
            raise HTTPClientError(
                f"failed to connect to {uri.host}:{uri.port}: {exc}"
            ) from exc

        reusable = False
        try:
            async with asyncio.timeout(self._read_timeout):
                connection.writer.write(wire)
                await connection.writer.drain()
                parsed = await self._read_final_response(
                    connection, request.method
                )
            reusable = (
                response_allows_reuse(parsed)
                and not connection.parser.has_buffered_data
            )
            return _to_http_response(parsed)
        except (TimeoutError, OSError) as exc:
            raise HTTPClientError(
                f"exchange with {uri.host}:{uri.port} failed: {exc}"
            ) from exc
        finally:
            await self._pool.release(connection, reusable=reusable)

    async def _open_connection(
        self, origin: Origin
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        return await open_upstream_connection(
            origin.host,
            origin.port,
            use_tls=origin.scheme == "https",
            verify=self._verify_tls,
        )

    def _create_parser(self) -> AsyncMultiResponseParser:
        return AsyncMultiResponseParser(max_read=self._max_read)

    async def _read_final_response(
        self,
        connection: PooledConnection,
        request_method: str,
    ) -> ParsedResponse:
        while True:
            parsed, _ = await connection.parser.next_response(
                connection.reader, request_method
            )
            if parsed is None:
                raise HTTPClientError("failed to parse upstream response")
            status = parsed.status_code or 0
            if status >= 200:
                return parsed
            LOG.debug(
                "Skipping 1xx informational response (%d) while "
                "waiting for the final response",
                status,
            )
