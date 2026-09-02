from __future__ import annotations

import pytest

from localstub.http.utils import (
    headers_to_headers,
    headers_to_message,
    message_from_items,
    serialize_header_line,
    status_phrase,
)


def test_headers_to_headers_decodes_ascii_values() -> None:
    headers = headers_to_headers([(b"Content-Type", b"application/json")])

    assert headers["Content-Type"] == "application/json"


def test_headers_to_headers_exposes_obs_text_through_raw_view() -> None:
    headers = headers_to_headers([(b"X-Obs", b"value-\x80\xff")])

    assert isinstance(headers["X-Obs"], str)
    assert headers.raw == ((b"X-Obs", b"value-\x80\xff"),)


def test_headers_to_message_preserves_string_facade() -> None:
    message = headers_to_message([(b"X-Obs", b"\x80\xff")])

    assert message["X-Obs"] == "\x80\xff"


def test_message_from_items_preserves_unicode_values() -> None:
    message = message_from_items([("X-Custom", "→")])

    assert message["X-Custom"] == "→"


def test_serialize_header_line_encodes_ascii_value() -> None:
    assert serialize_header_line("X-Custom", "value") == b"X-Custom: value\r\n"


def test_serialize_header_line_round_trips_obs_text_facade() -> None:
    line = serialize_header_line("X-Obs", "\x80\xff")

    assert line == b"X-Obs: \x80\xff\r\n"


def test_serialize_header_line_rejects_value_outside_byte_range() -> None:
    with pytest.raises(UnicodeEncodeError):
        serialize_header_line("X-Custom", "→")


def test_status_phrase_known_code_returns_phrase() -> None:
    assert status_phrase(200) == "OK"
    assert status_phrase(404) == "Not Found"


def test_status_phrase_unknown_code_returns_default() -> None:
    assert status_phrase(599, "UNKNOWN") == "UNKNOWN"


def test_status_phrase_unknown_code_without_default_returns_none() -> None:
    assert status_phrase(599) is None
