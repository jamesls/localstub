"""Unit tests for URI parsing utilities."""

from __future__ import annotations

import pytest

from localstub.http.request import HTTPRequest
from localstub.http.uri import parse_absolute_uri


class TestParseAbsoluteUri:
    """Tests for parse_absolute_uri function."""

    def test_parse_http_uri(self) -> None:
        result = parse_absolute_uri("http://example.com/path")
        assert result is not None
        assert result.scheme == "http"
        assert result.host == "example.com"
        assert result.port == 80
        assert result.path == "/path"

    def test_parse_https_uri(self) -> None:
        result = parse_absolute_uri("https://example.com/path")
        assert result is not None
        assert result.scheme == "https"
        assert result.host == "example.com"
        assert result.port == 443
        assert result.path == "/path"

    def test_parse_uri_with_explicit_port(self) -> None:
        result = parse_absolute_uri("http://example.com:8080/path")
        assert result is not None
        assert result.scheme == "http"
        assert result.host == "example.com"
        assert result.port == 8080
        assert result.path == "/path"

    def test_parse_uri_with_query_string(self) -> None:
        result = parse_absolute_uri("http://example.com/path?foo=bar&baz=qux")
        assert result is not None
        assert result.path == "/path?foo=bar&baz=qux"

    def test_parse_uri_with_no_path(self) -> None:
        result = parse_absolute_uri("http://example.com")
        assert result is not None
        assert result.path == "/"

    def test_parse_origin_form_returns_none(self) -> None:
        result = parse_absolute_uri("/path")
        assert result is None

    def test_parse_relative_path_returns_none(self) -> None:
        result = parse_absolute_uri("path/to/resource")
        assert result is None

    def test_parse_empty_string_returns_none(self) -> None:
        result = parse_absolute_uri("")
        assert result is None

    def test_parsed_uri_is_frozen(self) -> None:
        result = parse_absolute_uri("http://example.com/path")
        assert result is not None
        with pytest.raises(AttributeError):
            result.scheme = "https"  # type: ignore[misc]


class TestHTTPRequestProxyProperties:
    """Tests for HTTPRequest proxy-related properties."""

    def test_is_proxy_request_true_for_http(self) -> None:
        request = HTTPRequest(path="http://example.com/path")
        assert request.is_proxy_request is True

    def test_is_proxy_request_true_for_https(self) -> None:
        request = HTTPRequest(path="https://example.com/path")
        assert request.is_proxy_request is True

    def test_is_proxy_request_false_for_origin_form(self) -> None:
        request = HTTPRequest(path="/path")
        assert request.is_proxy_request is False

    def test_is_proxy_request_false_for_none_path(self) -> None:
        request = HTTPRequest(path=None)
        assert request.is_proxy_request is False

    def test_target_uri_returns_parsed_for_absolute(self) -> None:
        request = HTTPRequest(path="http://example.com:8080/api?key=val")
        uri = request.target_uri
        assert uri is not None
        assert uri.scheme == "http"
        assert uri.host == "example.com"
        assert uri.port == 8080
        assert uri.path == "/api?key=val"

    def test_target_uri_returns_none_for_origin_form(self) -> None:
        request = HTTPRequest(path="/path")
        assert request.target_uri is None

    def test_target_uri_returns_none_for_none_path(self) -> None:
        request = HTTPRequest(path=None)
        assert request.target_uri is None

    def test_effective_path_extracts_path_from_absolute(self) -> None:
        request = HTTPRequest(path="http://example.com/api/users?limit=10")
        assert request.effective_path == "/api/users?limit=10"

    def test_effective_path_returns_path_for_origin_form(self) -> None:
        request = HTTPRequest(path="/api/users")
        assert request.effective_path == "/api/users"

    def test_effective_path_returns_slash_for_none(self) -> None:
        request = HTTPRequest(path=None)
        assert request.effective_path == "/"
