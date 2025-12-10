from __future__ import annotations

from email.message import Message


def headers_to_message(headers: list[tuple[bytes, bytes]]) -> Message:
    """Convert parsed headers to email.message.Message for API compatibility.

    Args:
        headers: List of (name, value) byte tuples from the parser.

    Returns:
        Message object with headers accessible via dict-like interface.
    """
    msg = Message()
    for name, value in headers:
        msg[name.decode("iso-8859-1")] = value.decode("iso-8859-1")
    return msg
