from __future__ import annotations

import inspect
from email.message import Message
from http import HTTPStatus
from typing import Any, Awaitable, cast

from localstub.http.headers import Headers


def maybe_await(value: Any) -> Awaitable[Any]:
    """Wrap a sync-or-async return value into an awaitable."""
    if inspect.isawaitable(value):
        return cast(Awaitable[Any], value)

    async def done() -> Any:
        return value

    return done()


def headers_to_headers(headers: list[tuple[bytes, bytes]]) -> Headers:
    items: list[tuple[str, str]] = []
    for name, value in headers:
        items.append((
            name.decode("iso-8859-1"),
            value.decode("iso-8859-1"),
        ))
    return Headers.from_items(items)


def status_phrase(
    code: int,
    default: str | None = None,
) -> str | None:
    """Return the HTTP reason phrase for a status code.

    Returns *default* when the code is not a recognised HTTPStatus
    member.
    """
    try:
        return HTTPStatus(code).phrase
    except ValueError:
        return default


def decode_status_text(
    raw: bytes | None,
    default: str | None = None,
) -> str | None:
    """Decode raw status text bytes from an HTTP parser."""
    if raw is None:
        return default
    return raw.decode("ascii", errors="replace")


def headers_to_message(headers: list[tuple[bytes, bytes]]) -> Message:
    """Convert parsed headers to email.message.Message.

    Used for response recording/compatibility. Requests use Headers instead.
    """
    msg = Message()
    for name, value in headers:
        msg[name.decode("iso-8859-1")] = value.decode("iso-8859-1")
    return msg
