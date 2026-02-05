from __future__ import annotations

from email.message import Message

from localstub.http.headers import Headers


def headers_to_headers(headers: list[tuple[bytes, bytes]]) -> Headers:
    items: list[tuple[str, str]] = []
    for name, value in headers:
        items.append((
            name.decode("iso-8859-1"),
            value.decode("iso-8859-1"),
        ))
    return Headers.from_items(items)


def headers_to_message(headers: list[tuple[bytes, bytes]]) -> Message:
    """Convert parsed headers to email.message.Message.

    Used for response recording/compatibility. Requests use Headers instead.
    """
    msg = Message()
    for name, value in headers:
        msg[name.decode("iso-8859-1")] = value.decode("iso-8859-1")
    return msg
