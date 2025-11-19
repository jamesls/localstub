import pytest

from localstub.server import (
    AsyncHTTPTestServer,
    HTTPRequest,
    HTTPResponse,
)


def test_http_request_json_body_with_none_body():
    request = HTTPRequest(body=None)
    assert request.json_body is None


def test_http_request_json_body_with_empty_string():
    request = HTTPRequest(body="")
    assert request.json_body is None


def test_http_request_json_body_with_valid_json():
    request = HTTPRequest(body='{"key": "value"}')
    assert request.json_body == {"key": "value"}


def test_http_response_json_with_custom_headers():
    response = HTTPResponse.json(
        {"data": "test"},
        status=201,
        headers={"X-Custom": "value"},
    )
    assert response.status == 201
    assert response.headers["X-Custom"] == "value"
    assert response.headers["Content-Type"] == "application/json"
    assert "Content-Length" in response.headers


def test_http_response_text_with_custom_headers():
    response = HTTPResponse.text(
        "test text",
        status=202,
        headers={"X-Custom": "header"},
    )
    assert response.status == 202
    assert response.headers["X-Custom"] == "header"
    assert response.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert "Content-Length" in response.headers


def test_http_response_raw_with_custom_headers():
    response = HTTPResponse.raw(
        b"raw data",
        status=203,
        headers={"X-Custom": "raw"},
    )
    assert response.status == 203
    assert response.headers["X-Custom"] == "raw"
    assert "Content-Length" in response.headers


def test_server_url_raises_when_not_started():
    server = AsyncHTTPTestServer()
    with pytest.raises(RuntimeError, match="Server not started yet"):
        _ = server.url


def test_server_handler_getter():
    def handler(req):
        return HTTPResponse.json({"test": "value"})

    server = AsyncHTTPTestServer(handler=handler)
    assert server.handler is handler
