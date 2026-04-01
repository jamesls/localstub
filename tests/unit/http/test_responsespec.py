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
