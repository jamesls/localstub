from __future__ import annotations

import inspect
from collections.abc import Awaitable, Iterable
from email.message import Message
from http import HTTPStatus
from typing import cast

from localstub.http.headers import Headers

_STATUS_PHRASES: dict[int, str] = {
    status.value: status.phrase for status in HTTPStatus
}


def maybe_await[T](value: T | Awaitable[T]) -> Awaitable[T]:
    """Wrap a sync-or-async return value into an awaitable."""
    if inspect.isawaitable(value):
        return cast(Awaitable[T], value)

    async def done() -> T:
        return value

    return done()


def headers_to_headers(headers: list[tuple[bytes, bytes]]) -> Headers:
    return Headers.from_raw_items(headers)


def serialize_header_line(name: str, value: str) -> bytes:
    """Serialize a header from the reversible string facade."""
    return name.encode("ascii") + b": " + value.encode("latin-1") + b"\r\n"


def status_phrase(
    code: int,
    default: str | None = None,
) -> str | None:
    """Return the HTTP reason phrase for a status code.

    Returns *default* when the code is not a recognised HTTPStatus
    member.
    """
    return _STATUS_PHRASES.get(code, default)


def decode_status_text(
    raw: bytes | None,
    default: str | None = None,
) -> str | None:
    """Decode raw status text bytes from an HTTP parser."""
    if raw is None:
        return default
    return raw.decode("ascii", errors="replace")


def message_from_items(items: Iterable[tuple[str, str]]) -> Message:
    """Build an email.message.Message from decoded header pairs."""
    msg = Message()
    for name, value in items:
        msg[name] = value
    return msg


def headers_to_message(headers: list[tuple[bytes, bytes]]) -> Message:
    """Convert parsed headers through the reversible string facade."""
    return message_from_items(headers_to_headers(headers).items())
