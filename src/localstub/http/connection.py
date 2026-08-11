from __future__ import annotations

from email.message import Message

from localstub.http.headers import Headers
from localstub.http.request import RecordedHTTPRequest


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


def should_close_connection(
    request: RecordedHTTPRequest,
    *,
    response_headers: Headers | Message | dict[str, str] | None,
) -> bool:
    request_tokens = connection_tokens_from_headers(request.headers)

    if response_headers is None:
        response_tokens = set()
    elif isinstance(response_headers, dict):
        response_tokens = _connection_tokens_from_dict(response_headers)
    else:
        response_tokens = connection_tokens_from_headers(response_headers)

    if "close" in response_tokens:
        return True

    if _is_http11(request):
        return "close" in request_tokens

    if _is_http10(request):
        return "keep-alive" not in request_tokens

    return True
