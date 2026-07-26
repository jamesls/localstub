from __future__ import annotations

from localstub.http.headers import Headers
from localstub.http.proxy import build_origin_form_request
from localstub.http.request import HTTPRequest


def _proxy_request(uri: str) -> HTTPRequest:
    host = uri.split("://", 1)[1].split("/", 1)[0]
    return HTTPRequest(
        method="GET",
        path=uri,
        http_version="1.1",
        headers=Headers.from_items([("Host", host)]),
    )


def _host_header(wire: bytes) -> str:
    for line in wire.split(b"\r\n")[1:]:
        name, separator, value = line.partition(b":")
        if separator and name.strip().lower() == b"host":
            return value.strip().decode("ascii")
    raise AssertionError("no Host header in request")


def test_build_origin_form_request_keeps_non_default_port_for_scheme() -> None:
    request = _proxy_request("http://example.com:443/path")
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com:443"


def test_build_origin_form_request_keeps_port_80_for_https() -> None:
    request = _proxy_request("https://example.com:80/path")
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com:80"


def test_build_origin_form_request_omits_default_http_port() -> None:
    request = _proxy_request("http://example.com:80/path")
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com"


def test_build_origin_form_request_omits_default_https_port() -> None:
    request = _proxy_request("https://example.com:443/path")
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com"


def test_build_origin_form_request_adds_host_when_missing() -> None:
    request = HTTPRequest(
        method="GET",
        path="http://example.com:443/path",
        http_version="1.1",
    )
    uri = request.target_uri

    assert uri is not None

    wire = build_origin_form_request(request, uri)

    assert _host_header(wire) == "example.com:443"
