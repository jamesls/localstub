from __future__ import annotations

import pytest

from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest
from localstub.http.responsespec import HTTPResponse
from localstub.websocket.handshake import (
    HandshakeRequest,
    accept_headers,
    accept_key,
    header_tokens,
    validate_handshake,
)

RFC_KEY = "dGhlIHNhbXBsZSBub25jZQ=="
VALID_HEADERS = {
    "Host": "localhost",
    "Upgrade": "websocket",
    "Connection": "Upgrade",
    "Sec-WebSocket-Key": RFC_KEY,
    "Sec-WebSocket-Version": "13",
}


def _request(
    method: str = "GET",
    **overrides: str | None,
) -> HTTPRequest:
    items = dict(VALID_HEADERS)
    for name, value in overrides.items():
        header = name.replace("_", "-")
        if value is None:
            del items[header]
        else:
            items[header] = value
    return HTTPRequest(
        method=method, target="/", headers=Headers.from_items(items.items())
    )


def test_accept_key_matches_rfc_6455_section_1_3() -> None:
    assert accept_key(RFC_KEY) == "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="


def test_validate_returns_offer_for_valid_request() -> None:
    request = _request(
        Sec_WebSocket_Protocol="chat, superchat",
        Sec_WebSocket_Extensions="permessage-deflate",
    )

    offer = validate_handshake(request, "1.1")

    assert offer == HandshakeRequest(
        key=RFC_KEY,
        subprotocols=("chat", "superchat"),
        extensions=("permessage-deflate",),
    )


def test_validate_accepts_upgrade_token_among_others() -> None:
    request = _request(Connection="keep-alive, Upgrade", Upgrade="WebSocket")

    assert isinstance(validate_handshake(request, "1.1"), HandshakeRequest)


@pytest.mark.parametrize(
    ("request_", "version", "status", "header", "reason"),
    [
        (_request(Upgrade=None), "1.1", 426, "Upgrade", "Upgrade"),
        (_request(Upgrade="h2c"), "1.1", 426, "Upgrade", "Upgrade"),
        (
            _request(Sec_WebSocket_Version="8"),
            "1.1",
            426,
            "Sec-WebSocket-Version",
            "Version",
        ),
        (
            _request(Sec_WebSocket_Version=None),
            "1.1",
            426,
            "Sec-WebSocket-Version",
            "Version",
        ),
        (_request("POST"), "1.1", 400, None, "GET"),
        (_request(), "1.0", 400, None, "HTTP/1.1"),
        (_request(Connection="keep-alive"), "1.1", 400, None, "Connection"),
        (_request(Sec_WebSocket_Key=None), "1.1", 400, None, "Key"),
        (_request(Sec_WebSocket_Key="not base64!"), "1.1", 400, None, "Key"),
        (_request(Sec_WebSocket_Key="c2hvcnQ="), "1.1", 400, None, "Key"),
    ],
)
def test_validate_rejects_bad_handshake(
    request_: HTTPRequest,
    version: str,
    status: int,
    header: str | None,
    reason: str,
) -> None:
    response = validate_handshake(request_, version)

    assert isinstance(response, HTTPResponse)
    assert response.status == status
    if header is not None:
        assert header in response.headers
    assert reason in str(response.body)


def test_validate_rejects_two_keys() -> None:
    headers = Headers.from_items([
        *VALID_HEADERS.items(),
        ("Sec-WebSocket-Key", RFC_KEY),
    ])
    request = HTTPRequest(method="GET", target="/", headers=headers)

    response = validate_handshake(request, "1.1")

    assert isinstance(response, HTTPResponse)
    assert response.status == 400


def test_accept_headers_computes_101_headers() -> None:
    offer = HandshakeRequest(RFC_KEY, ("chat",), ())

    headers = accept_headers(
        offer,
        subprotocol="chat",
        extensions="permessage-deflate",
        extra=Headers.empty(),
    )

    assert list(headers.items()) == [
        ("Upgrade", "websocket"),
        ("Connection", "Upgrade"),
        ("Sec-WebSocket-Accept", "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="),
        ("Sec-WebSocket-Protocol", "chat"),
        ("Sec-WebSocket-Extensions", "permessage-deflate"),
    ]


def test_accept_headers_extra_replaces_computed_header_by_name() -> None:
    offer = HandshakeRequest(RFC_KEY, (), ())

    headers = accept_headers(
        offer,
        subprotocol=None,
        extensions=None,
        extra=Headers.from_items([
            ("sec-websocket-accept", "bogus"),
            ("X-Extra", "1"),
        ]),
    )

    assert headers.get_all("Sec-WebSocket-Accept") == ["bogus"]
    assert headers["X-Extra"] == "1"
    assert "Sec-WebSocket-Protocol" not in headers


def test_header_tokens_spans_repeated_headers() -> None:
    headers = Headers.from_items([("A", "x, y"), ("a", " , z")])

    assert header_tokens(headers, "A") == ["x", "y", "z"]
