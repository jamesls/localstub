from __future__ import annotations

import logging
from typing import cast

import httpx

from localstub.http.connection import connection_tokens_from_headers
from localstub.http.request import HTTPRequest
from localstub.http.responsespec import HTTPResponse
from localstub.http.uri import ParsedURI

LOG = logging.getLogger(__name__)

_DEFAULT_PORTS = {"http": 80, "https": 443}


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

    hop_by_hop = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
    headers: dict[str, str] = {}
    for name, value in request.headers.items():
        if name.lower() not in hop_by_hop:
            headers[name] = value

    try:
        upstream_response = await client.request(
            method=request.method or "GET",
            url=upstream_url,
            headers=headers,
            content=request.body_bytes or None,
        )
    except httpx.RequestError as exc:
        LOG.warning("Upstream request failed: %s", exc)
        return HTTPResponse(status=502, body=f"Bad Gateway: {exc}".encode())

    response_headers: dict[str, str] = {}
    for name, value in upstream_response.headers.items():
        if name.lower() not in hop_by_hop:
            response_headers[name] = value

    return HTTPResponse(
        status=upstream_response.status_code,
        headers=response_headers,
        body=cast(bytes, upstream_response.content),
    )
