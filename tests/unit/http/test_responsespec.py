from __future__ import annotations

from localstub.http.responsespec import HTTPResponse


def test_http_response_json_with_custom_headers() -> None:
    response = HTTPResponse.json(
        {"data": "test"},
        status=201,
        headers={"X-Custom": "value"},
    )

    assert response.status == 201
    assert response.headers["X-Custom"] == "value"
    assert response.headers["Content-Type"] == "application/json"
    assert "Content-Length" in response.headers


def test_http_response_text_with_custom_headers() -> None:
    response = HTTPResponse.text(
        "test text",
        status=202,
        headers={"X-Custom": "header"},
    )

    assert response.status == 202
    assert response.headers["X-Custom"] == "header"
    assert response.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert "Content-Length" in response.headers


def test_http_response_raw_with_custom_headers() -> None:
    response = HTTPResponse.raw(
        b"raw data",
        status=203,
        headers={"X-Custom": "raw"},
    )

    assert response.status == 203
    assert response.headers["X-Custom"] == "raw"
    assert "Content-Length" in response.headers


def test_http_response_raw_defaults_to_status_200() -> None:
    response = HTTPResponse.raw(b"hello")

    assert response.status == 200
    assert response.body == b"hello"
    assert response.headers["Content-Length"] == "5"


def test_http_response_preserves_iterable_headers_and_duplicates() -> None:
    items = [("Set-Cookie", "a=1"), ("Set-Cookie", "b=2")]

    response = HTTPResponse.text("hello", headers=iter(items))

    assert list(response.headers.items()) == [
        ("Content-Type", "text/plain; charset=utf-8"),
        ("Content-Length", "5"),
        *items,
    ]


def test_http_response_supplied_headers_override_defaults() -> None:
    response = HTTPResponse.text(
        "hello",
        headers={"content-type": "text/html", "CONTENT-LENGTH": "123"},
    )

    assert list(response.headers.items()) == [
        ("content-type", "text/html"),
        ("CONTENT-LENGTH", "123"),
    ]
