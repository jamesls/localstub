from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import overload

HeaderItem = tuple[str, str]


@dataclass(frozen=True)
class Headers:
    """Immutable, persistent HTTP headers.

    This type exists so request snapshots can be truly immutable while still
    supporting efficient request rewrites in middleware (structural sharing).

    Behavior:
    - Header name lookup is case-insensitive.
    - Duplicate headers are preserved for iteration and get_all().
    - patch_set() overrides all previous values for a header name.
    """

    _parent: Headers | None
    _items: tuple[HeaderItem, ...]
    _set_items: tuple[HeaderItem, ...]
    _set_lowers: frozenset[str]

    @classmethod
    def empty(cls) -> Headers:
        return _EMPTY_HEADERS

    @classmethod
    def from_items(cls, items: Iterable[HeaderItem]) -> Headers:
        return cls(
            _parent=None,
            _items=tuple(items),
            _set_items=(),
            _set_lowers=frozenset(),
        )

    def patch_set(self, values: Mapping[str, str]) -> Headers:
        if not values:
            return self

        lowered: dict[str, str] = {}
        set_items: list[HeaderItem] = []
        for name, value in values.items():
            name_lower = name.lower()
            if name_lower in lowered:
                raise ValueError(
                    "duplicate header name in patch_set (case-insensitive)"
                )
            lowered[name_lower] = name
            set_items.append((name, value))

        return Headers(
            _parent=self,
            _items=(),
            _set_items=tuple(set_items),
            _set_lowers=frozenset(lowered.keys()),
        )

    @overload
    def get_all(self, name: str) -> list[str] | None: ...

    @overload
    def get_all(self, name: str, failobj: list[str]) -> list[str]: ...

    @overload
    def get_all(self, name: str, failobj: None) -> list[str] | None: ...

    def get_all(
        self,
        name: str,
        failobj: list[str] | None = None,
    ) -> list[str] | None:
        name_lower = name.lower()

        if name_lower in self._set_lowers:
            values = [
                value
                for header_name, value in self._set_items
                if header_name.lower() == name_lower
            ]
            if values:
                return values
            return failobj

        if self._parent is not None:
            return self._parent.get_all(name, failobj=failobj)

        values = [
            value
            for header_name, value in self._items
            if header_name.lower() == name_lower
        ]
        if values:
            return values
        return failobj

    @overload
    def get(self, name: str) -> str | None: ...

    @overload
    def get(self, name: str, default: str) -> str: ...

    @overload
    def get(self, name: str, default: None) -> str | None: ...

    def get(self, name: str, default: str | None = None) -> str | None:
        values = self.get_all(name)
        if not values:
            return default
        return values[0]

    def __getitem__(self, name: str) -> str:
        values = self.get_all(name)
        if not values:
            raise KeyError(name)
        return values[0]

    def __contains__(self, name: str) -> bool:
        return bool(self.get_all(name))

    def items(self) -> Iterator[HeaderItem]:
        if self._parent is None:
            yield from self._items
            return

        for header_name, value in self._parent.items():
            if header_name.lower() in self._set_lowers:
                continue
            yield (header_name, value)

        yield from self._set_items


_EMPTY_HEADERS = Headers(
    _parent=None,
    _items=(),
    _set_items=(),
    _set_lowers=frozenset(),
)
