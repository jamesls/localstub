from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from localstub.http.headers import HeaderItem, Headers

type HeadersLike = Mapping[str, str] | Iterable[HeaderItem] | Headers


def normalize_headers(headers: HeadersLike | None) -> Headers:
    """Normalize any accepted headers shape to Headers."""
    if headers is None:
        return Headers.empty()
    if isinstance(headers, Headers):
        return headers
    if isinstance(headers, Mapping):
        # The isinstance check cannot distinguish the two Mapping-shaped
        # arms of HeadersLike, so re-anchor the element type explicitly.
        mapping = cast("Mapping[str, str]", headers)
        return Headers.from_items(mapping.items())
    return Headers.from_items(headers)


def _with_defaults(
    supplied: Headers,
    defaults: Iterable[HeaderItem],
) -> Headers:
    """Prepend default headers the caller didn't supply by name."""
    items = [item for item in defaults if item[0] not in supplied]
    items.extend(supplied.items())
    return Headers.from_items(items)


@dataclass
class HTTPResponse:
    status: int = 200
    headers: Headers = field(default_factory=Headers.empty)
    body: bytes | str = b""

    @classmethod
    def json(
        cls,
        obj: Any,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
    ) -> HTTPResponse:
        text = json.dumps(obj)
        body = text.encode("utf-8")
        return cls(
            status=status,
            headers=_with_defaults(
                normalize_headers(headers),
                (
                    ("Content-Type", "application/json"),
                    ("Content-Length", str(len(body))),
                ),
            ),
            body=body,
        )

    @classmethod
    def text(
        cls,
        text: str,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
    ) -> HTTPResponse:
        body = text.encode("utf-8")
        return cls(
            status=status,
            headers=_with_defaults(
                normalize_headers(headers),
                (
                    ("Content-Type", "text/plain; charset=utf-8"),
                    ("Content-Length", str(len(body))),
                ),
            ),
            body=body,
        )

    @classmethod
    def raw(
        cls,
        data: bytes,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
    ) -> HTTPResponse:
        return cls(
            status=status,
            headers=_with_defaults(
                normalize_headers(headers),
                (("Content-Length", str(len(data))),),
            ),
            body=data,
        )
