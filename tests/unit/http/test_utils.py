from __future__ import annotations

from localstub.http.utils import headers_to_message


def test_headers_to_message_empty_headers() -> None:
    message = headers_to_message([])

    assert len(message.keys()) == 0


def test_headers_to_message_single_header() -> None:
    message = headers_to_message([(b"Content-Type", b"application/json")])

    assert message["Content-Type"] == "application/json"


def test_headers_to_message_multiple_headers() -> None:
    headers = [
        (b"Host", b"example.com"),
        (b"Accept", b"*/*"),
        (b"Connection", b"keep-alive"),
    ]
    message = headers_to_message(headers)

    assert message["Host"] == "example.com"
    assert message["Accept"] == "*/*"
    assert message["Connection"] == "keep-alive"


def test_headers_to_message_decodes_iso_8859_1() -> None:
    message = headers_to_message([(b"X-Custom", b"\xe4\xf6\xfc")])

    assert message["X-Custom"] == "\xe4\xf6\xfc"
