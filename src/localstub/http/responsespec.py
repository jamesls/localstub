from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from localstub.http.headers import HeaderItem, Headers

type HeadersLike = Mapping[str, str] | Iterable[HeaderItem] | Headers


@dataclass(frozen=True)
class AbortTransmission:
    """Stop writing a response body and close the connection.

    Returned by a transmission strategy that stopped early, or yielded
    by a streamed body (``HTTPResponse.stream()``) to end the stream
    there.  The connection loop records the close with reason
    ``response_aborted``.
    """

    reset: bool = False
    """Request an abortive TCP close (RST) instead of a FIN."""


@dataclass(frozen=True)
class Pause:
    """Wait this many seconds before the next item of a streamed body."""

    seconds: float


type StreamItem = bytes | str | Pause | asyncio.Event | AbortTransmission
"""An item of a streamed body.

``bytes`` and ``str`` are sent as one chunk each.  A ``Pause`` waits
before the next item, an ``asyncio.Event`` waits until it is set, and
an ``AbortTransmission`` ends the stream and closes the connection.
"""

type StreamBody = Iterable[StreamItem] | AsyncIterable[StreamItem]

type SSEEvent = tuple[str | None, Any]
"""An ``(event_name, data)`` pair; a ``None`` name omits ``event:``."""


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


# The line endings an event stream parser recognizes.
_SSE_LINE_END = re.compile(r"\r\n|\r|\n")


def _encode_sse_event(name: str | None, data: Any) -> bytes:
    if name is not None and _SSE_LINE_END.search(name):
        raise ValueError(f"SSE event name contains a line break: {name!r}")
    text = data if isinstance(data, str) else json.dumps(data)
    lines = [] if name is None else [f"event: {name}"]
    lines.extend(f"data: {line}" for line in _SSE_LINE_END.split(text))
    return ("\n".join(lines) + "\n\n").encode("utf-8")


@dataclass
class HTTPResponse:
    status: int = 200
    headers: Headers = field(default_factory=Headers.empty)
    body: bytes | str = b""
    body_stream: StreamBody | None = None
    """Items sent as the body as they are produced, instead of ``body``.

    Sent with chunked framing when the headers say
    ``Transfer-Encoding: chunked``, as raw bytes when they carry a
    ``Content-Length``, and otherwise close-delimited: the connection
    closes after the last item.  The transmission strategy and
    ``write_chunk_size`` do not apply.  An async generator can be
    iterated only once, so return a fresh response from a handler for
    each request; a list can be reused.
    """

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

    @classmethod
    def stream(
        cls,
        items: StreamBody,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
        chunked: bool = True,
    ) -> HTTPResponse:
        """A response whose body is sent item by item as it is produced.

        *items* is an iterable or async iterable of ``StreamItem``.
        With ``chunked`` (and no ``Content-Length`` in *headers*) the
        body uses chunked transfer encoding; with ``chunked=False`` it
        is close-delimited.  A 1xx or 204 status, which has no body,
        gets no ``Transfer-Encoding`` (RFC 9112 §6.1).
        """
        supplied = normalize_headers(headers)
        defaults: tuple[HeaderItem, ...] = ()
        bodiless = 100 <= status < 200 or status == 204
        if chunked and not bodiless and "Content-Length" not in supplied:
            defaults = (("Transfer-Encoding", "chunked"),)
        return cls(
            status=status,
            headers=_with_defaults(supplied, defaults),
            body_stream=items,
        )

    @classmethod
    def sse(
        cls,
        events: Sequence[SSEEvent | Pause | asyncio.Event | AbortTransmission],
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
        chunked: bool = True,
    ) -> HTTPResponse:
        """A ``text/event-stream`` response sending one chunk per event.

        Each ``(event_name, data)`` pair becomes ``event:`` and
        ``data:`` lines; a ``str`` *data* is sent as is and anything
        else as JSON, one ``data:`` line per line of it.  ``Pause``,
        ``asyncio.Event`` and ``AbortTransmission`` items work as in
        ``stream()``.  The events are encoded up front, so the response
        can be reused.
        """
        items: list[StreamItem] = [
            _encode_sse_event(*event) if isinstance(event, tuple) else event
            for event in events
        ]
        return cls.stream(
            items,
            status=status,
            headers=_with_defaults(
                normalize_headers(headers),
                (
                    ("Content-Type", "text/event-stream"),
                    ("Cache-Control", "no-cache"),
                ),
            ),
            chunked=chunked,
        )
