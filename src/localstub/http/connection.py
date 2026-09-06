from __future__ import annotations

from email.message import Message

from localstub.http.headers import Headers
from localstub.http.request import RecordedHTTPRequest
from localstub.http.response import ParsedResponse


def parse_connection_tokens(value: str) -> set[str]:
    """Parse a Connection header value into lowercase tokens."""
    tokens: set[str] = set()
    for raw_token in value.split(","):
        token = raw_token.strip().lower()
        if token:
            tokens.add(token)
    return tokens


def connection_tokens_from_headers(
    headers: Headers | Message | None,
) -> set[str]:
    """Extract all Connection header tokens from headers."""
    if headers is None:
        return set()
    tokens: set[str] = set()
    for value in headers.get_all("Connection", []):
        tokens.update(parse_connection_tokens(value))
    return tokens


def _connection_tokens_from_dict(headers: dict[str, str]) -> set[str]:
    for name, value in headers.items():
        if name.lower() == "connection":
            return parse_connection_tokens(value)
    return set()


def _is_http10(request: RecordedHTTPRequest) -> bool:
    return request.http_version == "1.0"


def _is_http11(request: RecordedHTTPRequest) -> bool:
    return request.http_version == "1.1"


def response_allows_reuse(parsed: ParsedResponse) -> bool:
    """Return whether the connection may carry another request.

    The client-side sibling of ``should_close_connection``.  Reuse is
    ruled out by an incomplete or close-delimited response, a non-1.1
    HTTP version (HTTP/1.0 keep-alive is never reused), a
    ``Connection: close`` token, or a 101 protocol switch.
    """
    if not parsed.is_complete or parsed.is_eof_delimited:
        return False
    if parsed.http_version != "1.1":
        return False
    if parsed.status_code == 101:
        return False
    tokens: set[str] = set()
    for name, value in parsed.headers:
        if name.lower() == b"connection":
            tokens |= parse_connection_tokens(value.decode("latin-1"))
    return "close" not in tokens


def should_close_connection(
    request: RecordedHTTPRequest,
    *,
    response_headers: Headers | Message | dict[str, str] | None,
    response_version: str | None = None,
) -> bool:
    """Return whether the connection must close after this exchange.

    ``response_version`` is the HTTP version of the response placed on
    the wire.  Persistence is opt-in for HTTP/1.0 responses (RFC 9112
    §9.3); when the version is unknown (``None``) the response is
    assumed to be HTTP/1.1.
    """
    request_tokens = connection_tokens_from_headers(request.headers)

    if response_headers is None:
        response_tokens = set()
    elif isinstance(response_headers, dict):
        response_tokens = _connection_tokens_from_dict(response_headers)
    else:
        response_tokens = connection_tokens_from_headers(response_headers)

    if "close" in response_tokens:
        return True

    if response_version == "1.0" and "keep-alive" not in response_tokens:
        return True

    if _is_http11(request):
        return "close" in request_tokens

    if _is_http10(request):
        return "keep-alive" not in request_tokens

    return True
