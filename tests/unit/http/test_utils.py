from __future__ import annotations

from localstub.http.utils import (
    headers_to_message,
    message_from_items,
    status_phrase,
)


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


def test_message_from_items_empty_items() -> None:
    message = message_from_items([])

    assert len(message.keys()) == 0


def test_message_from_items_multiple_headers() -> None:
    items = [
        ("Content-Type", "application/json"),
        ("Content-Length", "12"),
    ]
    message = message_from_items(items)

    assert message["Content-Type"] == "application/json"
    assert message["Content-Length"] == "12"


def test_message_from_items_accepts_generator() -> None:
    message = message_from_items((name, "v") for name in ["A", "B"])

    assert message["A"] == "v"
    assert message["B"] == "v"


def test_message_from_items_preserves_non_latin_1_values() -> None:
    message = message_from_items([("X-Custom", "→")])

    assert message["X-Custom"] == "→"


def test_status_phrase_known_code_returns_phrase() -> None:
    assert status_phrase(200) == "OK"
    assert status_phrase(404) == "Not Found"


def test_status_phrase_unknown_code_returns_default() -> None:
    assert status_phrase(599, "UNKNOWN") == "UNKNOWN"


def test_status_phrase_unknown_code_without_default_returns_none() -> None:
    assert status_phrase(599) is None
