from __future__ import annotations

import logging
from dataclasses import replace

from localstub.http.client import HTTPClient, HTTPClientError
from localstub.http.connection import (
    connection_tokens_from_headers,
    parse_connection_tokens,
)
from localstub.http.headers import Headers
from localstub.http.request import RecordedHTTPRequest
from localstub.http.responsespec import HTTPResponse
from localstub.http.uri import ParsedURI

LOG = logging.getLogger(__name__)

_REQUEST_FRAMING_HEADERS = {"content-length", "transfer-encoding"}

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


def build_origin_form_request(
    recorded: RecordedHTTPRequest,
    uri: ParsedURI,
) -> bytes:
    """Convert absolute-form proxy request to origin-form for upstream."""
    path = recorded.effective_path or "/"
    method = recorded.method or "GET"
    version_value = recorded.http_version or "1.1"
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
    connection_tokens = connection_tokens_from_headers(recorded.headers)
    nominated_framing = connection_tokens & _REQUEST_FRAMING_HEADERS
    if nominated_framing:
        names = ", ".join(sorted(nominated_framing))
        raise ValueError(
            f"Connection header must not nominate request framing: {names}"
        )
    remove_headers = hop_by_hop | {"connection"} | connection_tokens
    authority = uri.authority
    host_added = False

    for name, value in recorded.headers.items():
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
    return header_bytes + recorded.wire_body_bytes


async def forward_proxy_request(
    client: HTTPClient,
    recorded: RecordedHTTPRequest,
) -> HTTPResponse:
    """Forward an absolute-form proxy request via *client*.

    The upstream request is built from the recorded request's value
    (which middleware may have rewritten) with hop-by-hop headers,
    Host, and Content-Length removed; the adapter regenerates framing.
    """
    uri = recorded.target_uri
    if uri is None:
        return HTTPResponse(
            status=400,
            body=b"Bad Request: Not an absolute URI",
        )

    drop_request_headers = (
        _HOP_BY_HOP_HEADERS
        | connection_tokens_from_headers(recorded.headers)
        | {"content-length", "host"}
    )
    request = replace(
        recorded.request,
        headers=Headers.from_items(
            (name, value)
            for name, value in recorded.headers.items()
            if name.lower() not in drop_request_headers
        ),
    )

    try:
        response = await client.send(request)
    except HTTPClientError as exc:
        LOG.warning("Upstream request failed: %s", exc)
        return HTTPResponse(status=502, body=f"Bad Gateway: {exc}".encode())

    drop_response_headers = set(_HOP_BY_HOP_HEADERS)
    for value in response.headers.get_all("Connection", []):
        drop_response_headers.update(parse_connection_tokens(value))
    return HTTPResponse(
        status=response.status,
        headers=Headers.from_items(
            (name, value)
            for name, value in response.headers.items()
            if name.lower() not in drop_response_headers
        ),
        body=response.body,
    )
