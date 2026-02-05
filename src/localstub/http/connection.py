from __future__ import annotations

from email.message import Message

from localstub.http.request import HTTPRequest


def _parse_connection_tokens(value: str) -> set[str]:
    tokens: set[str] = set()
    for raw_token in value.split(","):
        token = raw_token.strip().lower()
        if token:
            tokens.add(token)
    return tokens


def _connection_tokens_from_message(headers: Message | None) -> set[str]:
    if headers is None:
        return set()
    tokens: set[str] = set()
    for value in headers.get_all("Connection", []):
        tokens.update(_parse_connection_tokens(value))
    return tokens


def _connection_tokens_from_dict(headers: dict[str, str]) -> set[str]:
    for name, value in headers.items():
        if name.lower() == "connection":
            return _parse_connection_tokens(value)
    return set()


def _is_http10(request: HTTPRequest) -> bool:
    return request.http_version == "1.0"


def _is_http11(request: HTTPRequest) -> bool:
    return request.http_version == "1.1"


def should_close_connection(
    request: HTTPRequest,
    *,
    response_headers: Message | dict[str, str] | None,
) -> bool:
    request_tokens = _connection_tokens_from_message(request.headers)

    if response_headers is None:
        response_tokens = set()
    elif isinstance(response_headers, Message):
        response_tokens = _connection_tokens_from_message(response_headers)
    else:
        response_tokens = _connection_tokens_from_dict(response_headers)

    if "close" in response_tokens:
        return True

    if _is_http11(request):
        return "close" in request_tokens

    if _is_http10(request):
        return "keep-alive" not in request_tokens

    return True
