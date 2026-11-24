"""The opening handshake (RFC 6455 §4.2): validation and the 101.

``validate_handshake()`` checks an upgrade request and returns either
what the 101 needs or an ordinary rejection response whose body names
the failed check.  ``accept_headers()`` builds the 101's headers, with
caller-supplied headers replacing the computed ones by name.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from dataclasses import dataclass

from localstub.http.headers import HeaderItem, Headers
from localstub.http.request import HTTPRequest
from localstub.http.responsespec import HTTPResponse

ACCEPT_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
WEBSOCKET_VERSION = "13"


def accept_key(key: str) -> str:
    """The ``Sec-WebSocket-Accept`` value for a client's key."""
    digest = hashlib.sha1((key + ACCEPT_GUID).encode("latin-1")).digest()
    return base64.b64encode(digest).decode("ascii")


def header_tokens(headers: Headers, name: str) -> list[str]:
    """Comma-separated tokens across every ``name`` header, in order."""
    return [
        token.strip()
        for value in headers.get_all(name, [])
        for token in value.split(",")
        if token.strip()
    ]


@dataclass(frozen=True)
class HandshakeRequest:
    """What a valid upgrade request offers."""

    key: str
    subprotocols: tuple[str, ...]
    extensions: tuple[str, ...]
    """The raw ``Sec-WebSocket-Extensions`` header values."""


def _reject(
    status: int,
    reason: str,
    headers: dict[str, str] | None = None,
) -> HTTPResponse:
    return HTTPResponse.text(
        f"WebSocket handshake rejected: {reason}\n",
        status=status,
        headers=headers,
    )


def _valid_key(key: str) -> bool:
    try:
        decoded = base64.b64decode(key.encode("latin-1"), validate=True)
    except binascii.Error:
        return False
    return len(decoded) == 16


def validate_handshake(
    request: HTTPRequest,
    http_version: str,
) -> HandshakeRequest | HTTPResponse:
    """Check an upgrade request; return its offer or a rejection.

    A request without ``Upgrade: websocket`` or with a version other
    than 13 gets a 426 naming what the server speaks.  Any other
    failed check gets a 400.
    """
    headers = request.headers
    upgrades = {token.lower() for token in header_tokens(headers, "Upgrade")}
    if "websocket" not in upgrades:
        return _reject(
            426, "missing Upgrade: websocket", {"Upgrade": "websocket"}
        )
    versions = headers.get_all("Sec-WebSocket-Version", [])
    if versions != [WEBSOCKET_VERSION]:
        return _reject(
            426,
            f"Sec-WebSocket-Version must be 13, got {versions!r}",
            {"Sec-WebSocket-Version": WEBSOCKET_VERSION},
        )
    if request.method != "GET":
        return _reject(400, f"method must be GET, got {request.method}")
    if http_version != "1.1":
        return _reject(400, f"HTTP/1.1 required, got HTTP/{http_version}")
    connection = {
        token.lower() for token in header_tokens(headers, "Connection")
    }
    if "upgrade" not in connection:
        return _reject(400, "missing upgrade token in Connection")
    keys = headers.get_all("Sec-WebSocket-Key", [])
    if len(keys) != 1 or not _valid_key(keys[0].strip()):
        return _reject(400, "Sec-WebSocket-Key must be 16 bytes in base64")
    return HandshakeRequest(
        key=keys[0].strip(),
        subprotocols=tuple(header_tokens(headers, "Sec-WebSocket-Protocol")),
        extensions=tuple(headers.get_all("Sec-WebSocket-Extensions", [])),
    )


def accept_headers(
    handshake: HandshakeRequest,
    *,
    subprotocol: str | None,
    extensions: str | None,
    extra: Headers,
) -> Headers:
    """Headers of the 101; a name in ``extra`` replaces the computed one."""
    computed: list[HeaderItem] = [
        ("Upgrade", "websocket"),
        ("Connection", "Upgrade"),
        ("Sec-WebSocket-Accept", accept_key(handshake.key)),
    ]
    if subprotocol is not None:
        computed.append(("Sec-WebSocket-Protocol", subprotocol))
    if extensions is not None:
        computed.append(("Sec-WebSocket-Extensions", extensions))
    overridden = {name.lower() for name, _ in extra.raw}
    items = [
        item
        for item in computed
        if item[0].lower().encode("ascii") not in overridden
    ]
    items.extend(extra.items())
    return Headers.from_items(items)
