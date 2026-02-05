from __future__ import annotations

from email.message import Message


class FrozenMessage(Message):
    def _immutable(self) -> None:
        raise TypeError("headers are immutable; use with_headers()/rewrites")

    def __setitem__(self, name: str, val: str) -> None:
        self._immutable()

    def __delitem__(self, name: str) -> None:
        self._immutable()

    def add_header(self, *_args: object, **_kwargs: object) -> None:
        self._immutable()

    def replace_header(self, *_args: object, **_kwargs: object) -> None:
        self._immutable()

    def set_raw(self, *_args: object, **_kwargs: object) -> None:
        self._immutable()


def freeze_message(headers: Message) -> FrozenMessage:
    if isinstance(headers, FrozenMessage):
        return headers

    frozen = FrozenMessage()
    for name, value in headers.items():
        Message.__setitem__(frozen, name, value)
    return frozen


def headers_to_message(headers: list[tuple[bytes, bytes]]) -> Message:
    """Convert parsed headers to email.message.Message for API compatibility.

    Args:
        headers: List of (name, value) byte tuples from the parser.

    Returns:
        Message object with headers accessible via dict-like interface.
    """
    msg = FrozenMessage()
    for name, value in headers:
        Message.__setitem__(
            msg,
            name.decode("iso-8859-1"),
            value.decode("iso-8859-1"),
        )
    return msg
