from __future__ import annotations

import logging

import httpx

from localstub.http.connection import (
    connection_tokens_from_headers,
    parse_connection_tokens,
)
from localstub.http.request import HTTPRequest
from localstub.http.responsespec import HTTPResponse
from localstub.http.uri import ParsedURI

LOG = logging.getLogger(__name__)

_DEFAULT_PORTS = {"http": 80, "https": 443}

_HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
})


def _authority(uri: ParsedURI) -> str:
    """Render the Host header value for *uri*.

    The port is omitted only when it is the default for the scheme, so
    ``http://example.com:443/`` keeps its explicit port.
    """
    if uri.port == _DEFAULT_PORTS.get(uri.scheme):
        return uri.host
    return f"{uri.host}:{uri.port}"


def build_origin_form_request(request: HTTPRequest, uri: ParsedURI) -> bytes:
    """Convert absolute-form proxy request to origin-form for upstream."""
    path = request.effective_path or "/"
    method = request.method or "GET"
    version_value = request.http_version or "1.1"
    if version_value.startswith("HTTP/"):
        version = version_value
    else:
        version = f"HTTP/{version_value}"

    lines = [f"{method} {path} {version}"]

    hop_by_hop = {
        "proxy-connection",
        "proxy-authenticate",
        "proxy-authorization",
    }
    connection_tokens = connection_tokens_from_headers(request.headers)
    remove_headers = hop_by_hop | {"connection"} | connection_tokens
    authority = _authority(uri)
    host_added = False

    for name, value in request.headers.items():
        name_lower = name.lower()
        if name_lower == "host":
            lines.append(f"Host: {authority}")
            host_added = True
        elif name_lower in remove_headers:
            continue
        else:
            lines.append(f"{name}: {value}")

    if not host_added:
        lines.append(f"Host: {authority}")

    header_bytes = "\r\n".join(lines).encode("ascii") + b"\r\n\r\n"
    return header_bytes + request.wire_body_bytes


async def _read_upstream_body(
    upstream_response: httpx.Response,
) -> tuple[bytes, bool]:
    """Read the upstream body, returning it and whether it is decoded."""
    try:
        body = bytearray()
        async for chunk in upstream_response.aiter_raw():
            body.extend(chunk)
        return bytes(body), False
    except httpx.StreamConsumed:
        # A response hook (or a mock transport response built from
        # ``content=``) already consumed the raw stream. httpx caches
        # only the decoded body, so the upstream encoding and length
        # headers no longer describe it.
        return upstream_response.content, True


def _forwarded_response_headers(
    upstream_response: httpx.Response,
    *,
    body_is_decoded: bool,
) -> dict[str, str]:
    drop_headers = set(_HOP_BY_HOP_HEADERS)
    for value in upstream_response.headers.get_list("Connection"):
        drop_headers.update(parse_connection_tokens(value))
    if body_is_decoded:
        drop_headers.update({"content-encoding", "content-length"})

    response_headers: dict[str, str] = {}
    for name, value in upstream_response.headers.items():
        if name.lower() not in drop_headers:
            response_headers[name] = value
    return response_headers


async def forward_via_httpx(
    client: httpx.AsyncClient,
    request: HTTPRequest,
) -> HTTPResponse:
    """Forward an absolute-form proxy request using httpx."""
    uri = request.target_uri
    if uri is None:
        return HTTPResponse(
            status=400,
            body=b"Bad Request: Not an absolute URI",
        )

    upstream_url = request.path or "/"

    request_hop_by_hop = _HOP_BY_HOP_HEADERS | connection_tokens_from_headers(
        request.headers
    )
    headers: dict[str, str] = {}
    for name, value in request.headers.items():
        if name.lower() not in request_hop_by_hop:
            headers[name] = value

    try:
        async with client.stream(
            method=request.method or "GET",
            url=upstream_url,
            headers=headers,
            content=request.body_bytes or None,
        ) as upstream_response:
            body, body_is_decoded = await _read_upstream_body(
                upstream_response
            )
            return HTTPResponse(
                status=upstream_response.status_code,
                headers=_forwarded_response_headers(
                    upstream_response,
                    body_is_decoded=body_is_decoded,
                ),
                body=body,
            )
    except httpx.RequestError as exc:
        LOG.warning("Upstream request failed: %s", exc)
        return HTTPResponse(status=502, body=f"Bad Gateway: {exc}".encode())
