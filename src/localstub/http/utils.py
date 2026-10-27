from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import replace
from email.message import Message
from http import HTTPStatus
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

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
    return message_from_items(Headers.from_raw_items(headers).items())


class LazyBytes:
    """Bytes held as pieces and joined only when first read.

    Recorded wire bytes are mostly the same objects as a body that is
    already held, so keeping references instead of a joined copy
    stores each body once.
    """

    __slots__ = ("parts",)

    def __init__(self, parts: Iterable[bytes | memoryview]) -> None:
        self.parts = tuple(parts)

    def join(self) -> bytes:
        return b"".join(self.parts)


class _LazyBytesAttribute:
    """Data descriptor reading a ``bytes`` dataclass field lazily.

    The instance stores either ``bytes`` or ``LazyBytes`` under the
    field's own name; a read joins ``LazyBytes`` once and caches the
    result, so every read returns ``bytes``.
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __get__(self, obj: object | None, owner: type | None = None) -> bytes:
        if obj is None:
            # A field without a default has no class attribute.
            raise AttributeError(self._name)
        stored = vars(obj)
        value = stored[self._name]
        if isinstance(value, LazyBytes):
            value = value.join()
            stored[self._name] = value
        return value

    def __set__(self, obj: object, value: bytes) -> None:
        vars(obj)[self._name] = value


def lazy_bytes_fields[C: type](*names: str) -> Callable[[C], C]:
    """Make the named ``bytes`` fields of a dataclass readable lazily.

    Apply above ``@dataclass``.  The descriptors are attached after the
    dataclass is built, so its fields, constructor signature and
    generated methods are exactly those of plain ``bytes`` fields;
    ``set_lazy_bytes()`` then lets internal code store ``LazyBytes``.
    """

    def decorate(cls: C) -> C:
        for name in names:
            setattr(cls, name, _LazyBytesAttribute(name))
        return cls

    return decorate


def set_lazy_bytes(obj: object, name: str, value: bytes | LazyBytes) -> None:
    """Store *value* for a lazy field, bypassing the frozen guard."""
    vars(obj)[name] = value


def lazy_bytes_source(obj: object, name: str) -> bytes | LazyBytes:
    """Return what a lazy field holds without joining it."""
    return vars(obj)[name]


def replace_keeping_lazy[T: DataclassInstance](
    obj: T, name: str, **changes: Any
) -> T:
    """``dataclasses.replace()`` that carries field *name* over unjoined."""
    replaced = replace(obj, **{name: b""}, **changes)
    set_lazy_bytes(replaced, name, lazy_bytes_source(obj, name))
    return replaced
