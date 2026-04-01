from __future__ import annotations

import pytest

from localstub.http.uri import parse_absolute_uri


def test_parse_absolute_uri_http() -> None:
    result = parse_absolute_uri("http://example.com/path")

    assert result is not None
    assert result.scheme == "http"
    assert result.host == "example.com"
    assert result.port == 80
    assert result.path == "/path"


def test_parse_absolute_uri_https() -> None:
    result = parse_absolute_uri("https://example.com/path")

    assert result is not None
    assert result.scheme == "https"
    assert result.host == "example.com"
    assert result.port == 443
    assert result.path == "/path"


def test_parse_absolute_uri_with_explicit_port() -> None:
    result = parse_absolute_uri("http://example.com:8080/path")

    assert result is not None
    assert result.scheme == "http"
    assert result.host == "example.com"
    assert result.port == 8080
    assert result.path == "/path"


def test_parse_absolute_uri_with_query_string() -> None:
    result = parse_absolute_uri("http://example.com/path?foo=bar&baz=qux")

    assert result is not None
    assert result.path == "/path?foo=bar&baz=qux"


def test_parse_absolute_uri_with_no_path() -> None:
    result = parse_absolute_uri("http://example.com")

    assert result is not None
    assert result.path == "/"


def test_parse_absolute_uri_origin_form_returns_none() -> None:
    assert parse_absolute_uri("/path") is None


def test_parse_absolute_uri_relative_path_returns_none() -> None:
    assert parse_absolute_uri("path/to/resource") is None


def test_parse_absolute_uri_empty_string_returns_none() -> None:
    assert parse_absolute_uri("") is None


def test_parsed_uri_is_frozen() -> None:
    result = parse_absolute_uri("http://example.com/path")

    assert result is not None

    with pytest.raises(AttributeError):
        result.scheme = "https"
