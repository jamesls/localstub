from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.http.uri import ParsedURI, parse_absolute_uri


def _uri_like(scheme: str, host: str, port: int | str, path: str) -> str:
    return f"{scheme}://{host}:{port}{path}"


URI_LIKE_STRINGS = st.one_of(
    st.text(),
    st.builds(
        _uri_like,
        st.sampled_from(["http", "https"]),
        st.text(),
        st.one_of(st.integers(), st.text()),
        st.text(),
    ),
)


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


def test_parsed_uri_authority_omits_default_http_port() -> None:
    result = parse_absolute_uri("http://example.com/path")

    assert result is not None
    assert result.authority == "example.com"


def test_parsed_uri_authority_omits_default_https_port() -> None:
    result = parse_absolute_uri("https://example.com/path")

    assert result is not None
    assert result.authority == "example.com"


def test_parsed_uri_authority_keeps_explicit_port() -> None:
    result = parse_absolute_uri("http://example.com:8080/path")

    assert result is not None
    assert result.authority == "example.com:8080"


def test_parsed_uri_authority_keeps_other_schemes_default_port() -> None:
    result = parse_absolute_uri("http://example.com:443/path")

    assert result is not None
    assert result.authority == "example.com:443"


def test_parsed_uri_authority_brackets_ipv6_host_with_port() -> None:
    result = parse_absolute_uri("http://[::1]:8080/path")

    assert result is not None
    assert result.host == "::1"
    assert result.authority == "[::1]:8080"


def test_parsed_uri_authority_brackets_ipv6_host_default_port() -> None:
    result = parse_absolute_uri("http://[::1]/path")

    assert result is not None
    assert result.authority == "[::1]"


@given(uri=URI_LIKE_STRINGS)
def test_parse_absolute_uri_with_arbitrary_text_returns_result_or_none(
    uri: str,
) -> None:
    result = parse_absolute_uri(uri)

    assert result is None or isinstance(result, ParsedURI)


def test_parsed_uri_is_frozen() -> None:
    result = parse_absolute_uri("http://example.com/path")

    assert result is not None

    with pytest.raises(AttributeError):
        result.scheme = "https"
