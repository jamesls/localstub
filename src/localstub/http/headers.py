from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import overload

HeaderItem = tuple[str, str]
RawHeaderItem = tuple[bytes, bytes]


def _encode_name(name: str) -> bytes:
    try:
        return name.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError(f"header name must be ASCII: {name!r}") from None


def _encode_value(value: str) -> bytes:
    try:
        return value.encode("latin-1")
    except UnicodeEncodeError:
        raise ValueError(
            f"header value must be within the Latin-1 range: {value!r}"
        ) from None


def _encode_item(item: HeaderItem) -> RawHeaderItem:
    name, value = item
    return _encode_name(name), _encode_value(value)


def _validate_raw_item(item: RawHeaderItem) -> RawHeaderItem:
    name, _ = item
    if not name.isascii():
        raise ValueError(f"header name must be ASCII: {name!r}")
    return item


def _decode_item(item: RawHeaderItem) -> HeaderItem:
    name, value = item
    return name.decode("ascii"), value.decode("latin-1")


@dataclass(frozen=True, eq=False)
class Headers:
    """Immutable, persistent HTTP headers.

    This type exists so request snapshots can be truly immutable while still
    supporting efficient request rewrites in middleware (structural sharing).

    Behavior:
    - Header name lookup is case-insensitive.
    - Lookups always return strings, using Latin-1 as a reversible facade.
    - Lookups by a non-ASCII name never match; such a name cannot exist.
    - Constructing with a non-ASCII name or a value outside the Latin-1
      range raises ValueError.
    - raw exposes the underlying byte pairs without interpreting obs-text.
    - Duplicate headers are preserved for iteration and get_all().
    - patch_set() overrides all previous values for a header name.
    - Equality is value equality over items(); two header sets with the
      same items compare equal regardless of patch history.
    """

    _parent: Headers | None
    _items: tuple[RawHeaderItem, ...]
    _set_items: tuple[RawHeaderItem, ...]
    _set_lowers: frozenset[bytes]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Headers):
            return NotImplemented
        return self.raw == other.raw

    def __hash__(self) -> int:
        return hash(self.raw)

    @classmethod
    def empty(cls) -> Headers:
        return _EMPTY_HEADERS

    @classmethod
    def from_items(cls, items: Iterable[HeaderItem]) -> Headers:
        return cls(
            _parent=None,
            _items=tuple(_encode_item(item) for item in items),
            _set_items=(),
            _set_lowers=frozenset(),
        )

    @classmethod
    def from_raw_items(cls, items: Iterable[RawHeaderItem]) -> Headers:
        """Create headers from uninterpreted wire-level byte pairs."""
        return cls(
            _parent=None,
            _items=tuple(_validate_raw_item(item) for item in items),
            _set_items=(),
            _set_lowers=frozenset(),
        )

    def patch_set(self, values: Mapping[str, str]) -> Headers:
        if not values:
            return self

        lowered: set[bytes] = set()
        set_items: list[RawHeaderItem] = []
        for name, value in values.items():
            name_bytes = _encode_name(name)
            name_lower = name_bytes.lower()
            if name_lower in lowered:
                raise ValueError(
                    "duplicate header name in patch_set (case-insensitive)"
                )
            lowered.add(name_lower)
            set_items.append((name_bytes, _encode_value(value)))

        return Headers(
            _parent=self,
            _items=(),
            _set_items=tuple(set_items),
            _set_lowers=frozenset(lowered),
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
        if not name.isascii():
            return failobj
        name_lower = name.lower().encode("ascii")

        if name_lower in self._set_lowers:
            # patch_set only records a lowered name alongside an item
            # carrying it, so this list is never empty.
            return [
                value.decode("latin-1")
                for header_name, value in self._set_items
                if header_name.lower() == name_lower
            ]

        if self._parent is not None:
            return self._parent.get_all(name, failobj=failobj)

        values = [
            value.decode("latin-1")
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

    def _raw_items(self) -> Iterator[RawHeaderItem]:
        if self._parent is None:
            yield from self._items
            return

        for header_name, value in self._parent.raw:
            if header_name.lower() in self._set_lowers:
                continue
            yield (header_name, value)

        yield from self._set_items

    @property
    def raw(self) -> tuple[RawHeaderItem, ...]:
        """Return the uninterpreted wire-level header byte pairs."""
        return tuple(self._raw_items())

    def items(self) -> Iterator[HeaderItem]:
        for item in self._raw_items():
            yield _decode_item(item)


_EMPTY_HEADERS = Headers(
    _parent=None,
    _items=(),
    _set_items=(),
    _set_lowers=frozenset(),
)
