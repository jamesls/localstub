"""Hypothesis strategies for generating HTTP wire bytes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

from hypothesis import strategies as st

TOKEN_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-"
HEADER_VALUE_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 -_.=/"
_EXTENSION_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"
RESERVED_HEADER_NAMES = frozenset({
    "connection",
    "content-length",
    "expect",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
})


@dataclass(frozen=True)
class ChunkedBody:
    """A chunked transfer encoding of ``payloads``.

    ``encoded`` is a complete chunked body: size lines (with random
    hex-digit case and optional chunk extensions), chunk data, the
    terminal chunk, optional trailers, and the final CRLF.
    """

    payloads: list[bytes]
    encoded: bytes


def header_item() -> st.SearchStrategy[tuple[str, str]]:
    """A header (name, value) pair that never affects message framing."""
    names = st.text(
        alphabet=TOKEN_ALPHABET,
        min_size=1,
        max_size=10,
    ).filter(lambda name: name.lower() not in RESERVED_HEADER_NAMES)
    values = st.text(alphabet=HEADER_VALUE_ALPHABET, max_size=12)
    return st.tuples(names, values)


def header_items() -> st.SearchStrategy[list[tuple[str, str]]]:
    return st.lists(header_item(), max_size=4)


def obs_text_value() -> st.SearchStrategy[bytes]:
    """A non-empty field value containing only RFC 9110 obs-text octets."""
    return st.lists(
        st.integers(min_value=0x80, max_value=0xFF),
        min_size=1,
        max_size=32,
    ).map(bytes)


@st.composite
def chunked_bodies(draw: st.DrawFn) -> ChunkedBody:
    def chunk_size_line(size: int) -> bytes:
        line = format(size, draw(st.sampled_from("xX"))).encode("ascii")
        if draw(st.booleans()):
            name = draw(
                st.text(alphabet=_EXTENSION_ALPHABET, min_size=1, max_size=6)
            )
            line += b";" + name.encode("ascii")
            if draw(st.booleans()):
                value = draw(
                    st.text(
                        alphabet=_EXTENSION_ALPHABET, min_size=1, max_size=6
                    )
                )
                line += b"=" + value.encode("ascii")
        return line + b"\r\n"

    payloads = draw(st.lists(st.binary(min_size=1, max_size=32), max_size=4))
    parts = [
        chunk_size_line(len(payload)) + payload + b"\r\n"
        for payload in payloads
    ]
    parts.append(chunk_size_line(0))
    parts.extend(
        f"{name}: {value}\r\n".encode("ascii")
        for name, value in draw(st.lists(header_item(), max_size=2))
    )
    parts.append(b"\r\n")
    return ChunkedBody(payloads=payloads, encoded=b"".join(parts))


@st.composite
def fragments_of(draw: st.DrawFn, data: bytes) -> list[bytes]:
    """Partition ``data`` into consecutive non-empty fragments."""
    if len(data) < 2:
        return [data] if data else []
    cuts = draw(
        st.lists(
            st.integers(min_value=1, max_value=len(data) - 1),
            unique=True,
            max_size=8,
        ).map(sorted)
    )
    bounds = [0, *cuts, len(data)]
    return [data[start:end] for start, end in pairwise(bounds)]
