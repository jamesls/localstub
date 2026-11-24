"""permessage-deflate (RFC 7692): negotiation and per-message codecs.

``negotiate()`` picks the first offer in a client's
``Sec-WebSocket-Extensions`` that the server can honour and returns
the agreed parameters with the response header value.
``agreed_extensions()`` reads the agreement back from a 101, which is
how a session learns what it told the client and how the forward
relay learns what an upstream agreed.

``MessageDeflater`` and ``MessageInflater`` compress and inflate one
message at a time, keeping a compression context across messages
unless ``no_context_takeover`` was agreed.  They use the standard
library's ``zlib``.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterable
from dataclasses import dataclass

EXTENSION_NAME = "permessage-deflate"
# Appended before inflating and stripped after deflating (RFC 7692
# §7.2.1): the empty stored block a sync flush ends with.
SYNC_TAIL = b"\x00\x00\xff\xff"
DEFAULT_WINDOW_BITS = 15
_MIN_WINDOW_BITS = 8
_MAX_WINDOW_BITS = 15
# zlib 1.2.9 and later refuse a 256-byte window for raw deflate, so a
# compressor cannot honour a window of 8 bits.
_MIN_DEFLATE_WINDOW_BITS = 9

type ExtensionParams = list[tuple[str, str | None]]
type ExtensionOffer = tuple[str, ExtensionParams]


@dataclass(frozen=True)
class PerMessageDeflate:
    """Server preferences for permessage-deflate (RFC 7692).

    The same type holds the parameters a handshake agreed, as
    ``RecordedWebSocket.compression``.
    """

    server_no_context_takeover: bool = False
    client_no_context_takeover: bool = False
    server_max_window_bits: int | None = None
    client_max_window_bits: int | None = None

    def __post_init__(self) -> None:
        for name in ("server_max_window_bits", "client_max_window_bits"):
            bits: int | None = getattr(self, name)
            if bits is not None and not (
                _MIN_WINDOW_BITS <= bits <= _MAX_WINDOW_BITS
            ):
                raise ValueError(f"{name} must be between 8 and 15")

    def header_value(self) -> str:
        """The ``Sec-WebSocket-Extensions`` value naming these params."""
        parts = [EXTENSION_NAME]
        if self.server_no_context_takeover:
            parts.append("server_no_context_takeover")
        if self.client_no_context_takeover:
            parts.append("client_no_context_takeover")
        if self.server_max_window_bits is not None:
            parts.append(
                f"server_max_window_bits={self.server_max_window_bits}"
            )
        if self.client_max_window_bits is not None:
            parts.append(
                f"client_max_window_bits={self.client_max_window_bits}"
            )
        return "; ".join(parts)


def _split_outside_quotes(value: str, separator: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    quoted = False
    for char in value:
        if char == '"':
            quoted = not quoted
        if char == separator and not quoted:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return parts


def parse_extensions(values: Iterable[str]) -> list[ExtensionOffer]:
    """Parse ``Sec-WebSocket-Extensions`` header values in order.

    Each element is an extension name with its parameters, in the
    order listed; quoted parameter values are unquoted.  Empty list
    elements are skipped, as RFC 9110 list syntax allows.
    """
    offers: list[ExtensionOffer] = []
    for value in values:
        for element in _split_outside_quotes(value, ","):
            pieces = [
                piece.strip() for piece in _split_outside_quotes(element, ";")
            ]
            if not pieces[0]:
                continue
            params: ExtensionParams = []
            for piece in pieces[1:]:
                name, sep, param_value = piece.partition("=")
                name = name.strip().lower()
                if not sep:
                    params.append((name, None))
                    continue
                param_value = param_value.strip()
                if (
                    len(param_value) >= 2
                    and param_value[0] == '"' == (param_value[-1])
                ):
                    param_value = param_value[1:-1]
                params.append((name, param_value))
            offers.append((pieces[0].lower(), params))
    return offers


def _window_bits(value: str | None) -> int | None:
    """Parse a window-bits value; ``None`` when it is malformed."""
    if value is None or not value.isdigit() or value != str(int(value)):
        return None
    bits = int(value)
    if _MIN_WINDOW_BITS <= bits <= _MAX_WINDOW_BITS:
        return bits
    return None


_FLAG_PARAMS = frozenset({
    "server_no_context_takeover",
    "client_no_context_takeover",
})


@dataclass(frozen=True)
class _Offer:
    server_no_context_takeover: bool = False
    client_no_context_takeover: bool = False
    server_max_window_bits: int | None = None
    client_max_window_bits_named: bool = False
    client_max_window_bits: int | None = None


# Stands for a parameter that carries no value.
_NO_VALUE = 0


def _parse_param(
    name: str, value: str | None, *, response: bool
) -> int | None:
    """Parse one parameter's value; ``None`` when it is invalid.

    Flags and a ``client_max_window_bits`` named without a value, which
    only a client offer may do, parse as ``_NO_VALUE``.
    """
    if name in _FLAG_PARAMS:
        return _NO_VALUE if value is None else None
    if name == "client_max_window_bits" and value is None and not response:
        return _NO_VALUE
    if name in {"server_max_window_bits", "client_max_window_bits"}:
        return _window_bits(value)
    return None


def _parse_params(params: ExtensionParams, *, response: bool) -> _Offer | None:
    """Read permessage-deflate parameters; ``None`` if any is invalid.

    An unknown or repeated parameter, or a malformed value, makes the
    whole offer (or agreement) invalid (RFC 7692 §7.1).
    """
    values: dict[str, int] = {}
    for name, value in params:
        if name in values:
            return None
        parsed = _parse_param(name, value, response=response)
        if parsed is None:
            return None
        values[name] = parsed
    client_bits = values.get("client_max_window_bits")
    return _Offer(
        server_no_context_takeover="server_no_context_takeover" in values,
        client_no_context_takeover="client_no_context_takeover" in values,
        server_max_window_bits=values.get("server_max_window_bits"),
        client_max_window_bits_named=client_bits is not None,
        client_max_window_bits=client_bits or None,
    )


def _smaller(first: int | None, second: int | None) -> int | None:
    if first is None:
        return second
    if second is None:
        return first
    return min(first, second)


def negotiate(
    offer_values: Iterable[str],
    preferences: PerMessageDeflate,
) -> PerMessageDeflate | None:
    """Agree permessage-deflate from a client's offers, or decline.

    The first offer the server can honour wins.  Offers with an
    unknown or repeated parameter are skipped, and so is an offer that
    requires ``server_max_window_bits=8``, which zlib cannot honour.
    """
    for name, params in parse_extensions(offer_values):
        if name != EXTENSION_NAME:
            continue
        offer = _parse_params(params, response=False)
        if offer is None:
            continue
        if offer.server_max_window_bits == _MIN_WINDOW_BITS:
            continue
        server_bits = _smaller(
            offer.server_max_window_bits, preferences.server_max_window_bits
        )
        if server_bits is not None and server_bits < _MIN_DEFLATE_WINDOW_BITS:
            continue
        client_bits = None
        if offer.client_max_window_bits_named:
            client_bits = _smaller(
                offer.client_max_window_bits,
                preferences.client_max_window_bits,
            )
        return PerMessageDeflate(
            server_no_context_takeover=(
                offer.server_no_context_takeover
                or preferences.server_no_context_takeover
            ),
            client_no_context_takeover=(
                offer.client_no_context_takeover
                or preferences.client_no_context_takeover
            ),
            server_max_window_bits=server_bits,
            client_max_window_bits=client_bits,
        )
    return None


@dataclass(frozen=True)
class AgreedExtensions:
    """What a 101's ``Sec-WebSocket-Extensions`` agreed.

    ``compression`` is the permessage-deflate agreement, if any.
    ``other`` is ``True`` when the header names any other extension,
    whose use of the reserved bits localstub cannot interpret.
    """

    compression: PerMessageDeflate | None = None
    other: bool = False


def agreed_extensions(values: Iterable[str]) -> AgreedExtensions:
    """Read the extensions a 101 response agreed."""
    compression: PerMessageDeflate | None = None
    other = False
    for name, params in parse_extensions(values):
        if name != EXTENSION_NAME or compression is not None:
            other = True
            continue
        parsed = _parse_params(params, response=True)
        if parsed is None:
            other = True
            continue
        compression = PerMessageDeflate(
            server_no_context_takeover=parsed.server_no_context_takeover,
            client_no_context_takeover=parsed.client_no_context_takeover,
            server_max_window_bits=parsed.server_max_window_bits,
            client_max_window_bits=parsed.client_max_window_bits,
        )
    return AgreedExtensions(compression=compression, other=other)


class InflateError(Exception):
    """A compressed message did not inflate."""


class MessageTooBig(Exception):
    """An inflated message grew past its size limit."""


class MessageDeflater:
    """Compress one message at a time (RFC 7692 §7.2.1)."""

    def __init__(
        self,
        *,
        no_context_takeover: bool,
        max_window_bits: int | None,
    ) -> None:
        self._no_context_takeover = no_context_takeover
        self._window_bits = max_window_bits or DEFAULT_WINDOW_BITS
        self._compressor = self._new_compressor()
        self._used = False

    def _new_compressor(self) -> zlib._Compress:
        return zlib.compressobj(
            zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -self._window_bits
        )

    def compress(self, data: bytes) -> bytes:
        if self._used and self._no_context_takeover:
            self._compressor = self._new_compressor()
        self._used = True
        compressor = self._compressor
        out = compressor.compress(data) + compressor.flush(zlib.Z_SYNC_FLUSH)
        return out.removesuffix(SYNC_TAIL)


class MessageInflater:
    """Inflate one message at a time, fed frame by frame.

    ``start()`` begins a message.  ``feed()`` inflates the payload of
    each frame as it arrives, so a size limit is enforced while
    inflating rather than after: a small compressed message cannot
    expand without bound.  ``finish()`` appends the sync tail that
    ends the message.
    """

    def __init__(
        self,
        *,
        no_context_takeover: bool,
        max_window_bits: int | None,
    ) -> None:
        self._no_context_takeover = no_context_takeover
        self._window_bits = max_window_bits or DEFAULT_WINDOW_BITS
        self._decompressor = zlib.decompressobj(-self._window_bits)
        self._used = False
        self._size = 0

    def start(self) -> None:
        """Begin a new message."""
        self._size = 0
        if self._used and self._no_context_takeover:
            self._decompressor = zlib.decompressobj(-self._window_bits)
        self._used = True

    def feed(self, data: bytes, limit: int | None) -> bytes:
        """Inflate ``data``; raise once the message passes ``limit``."""
        decompressor = self._decompressor
        try:
            if limit is None:
                out = decompressor.decompress(data)
            else:
                room = limit - self._size
                out = decompressor.decompress(data, room + 1)
                if len(out) > room:
                    raise MessageTooBig
        except zlib.error as exc:
            raise InflateError(str(exc)) from exc
        self._size += len(out)
        return out

    def finish(self, limit: int | None) -> bytes:
        """Inflate the sync tail that ends the message."""
        return self.feed(SYNC_TAIL, limit)
