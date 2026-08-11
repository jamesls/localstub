from __future__ import annotations

import base64
import io
import json
from datetime import UTC, datetime

import pytest

from localstub.http.exchange import RecordedExchange
from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HTTPResponse
from localstub.server import AsyncHTTPTestServer
from localstub.traffic_jsonl import (
    JSONLTrafficWriter,
    dump_server_traffic_jsonl,
    exchange_to_json_obj,
)


def _recorded_request(
    method: str,
    target: str,
    *,
    headers: Headers | None = None,
    body: bytes | None = None,
    wire_raw_bytes: bytes = b"",
    client: tuple[str, int] | None = None,
) -> RecordedHTTPRequest:
    request = HTTPRequest(
        method=method,
        target=target,
        headers=headers if headers is not None else Headers.empty(),
        body=body,
    )
    return RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=wire_raw_bytes,
        http_version="1.1",
        client=client,
    )


def test_exchange_to_json_obj_encodes_wire_bytes_and_timestamps() -> None:
    request = _recorded_request(
        "GET",
        "/example",
        headers=Headers.from_items([("Host", "example.com")]),
        body=b"",
        wire_raw_bytes=b"GET /example HTTP/1.1\r\n\r\n",
        client=("127.0.0.1", 12345),
    )

    response = RecordedHTTPResponse(
        response=HTTPResponse(
            status=200,
            headers=Headers.from_items([("Content-Type", "text/plain")]),
            body=b"hello",
        ),
        reason="OK",
        wire_raw_bytes=b"HTTP/1.1 200 OK\r\n\r\nhello",
    )

    request_timestamp = datetime(2026, 1, 27, 12, 0, 0, tzinfo=UTC)
    response_timestamp = datetime(2026, 1, 27, 12, 0, 1, tzinfo=UTC)
    exchange = RecordedExchange(
        request=request,
        response=response,
        request_timestamp=request_timestamp,
        response_timestamp=response_timestamp,
    )

    obj = exchange_to_json_obj(exchange)
    assert obj["timestamp"] == request_timestamp.isoformat()
    assert obj["response_timestamp"] == response_timestamp.isoformat()

    req_obj = obj["request"]
    assert isinstance(req_obj, dict)
    assert req_obj["method"] == "GET"
    assert req_obj["path"] == "/example"
    assert req_obj["client"] == {"host": "127.0.0.1", "port": 12345}
    assert req_obj["headers"] == {"Host": "example.com"}
    assert req_obj["body"] == ""
    assert (
        base64.b64decode(req_obj["raw_wire_bytes"])
        == b"GET /example HTTP/1.1\r\n\r\n"
    )

    resp_obj = obj["response"]
    assert isinstance(resp_obj, dict)
    assert resp_obj["status"] == 200
    assert resp_obj["reason"] == "OK"
    assert resp_obj["headers"] == {"Content-Type": "text/plain"}
    assert resp_obj["body"] == "hello"
    assert (
        base64.b64decode(resp_obj["raw_wire_bytes"])
        == b"HTTP/1.1 200 OK\r\n\r\nhello"
    )


def test_exchange_to_json_obj_handles_decode_errors() -> None:
    request = _recorded_request(
        "POST",
        "/binary",
        body=b"\xff",
        wire_raw_bytes=b"POST /binary HTTP/1.1\r\n\r\n\xff",
    )
    response = RecordedHTTPResponse(
        response=HTTPResponse(status=200),
        reason="OK",
        wire_raw_bytes=b"HTTP/1.1 200 OK\r\n\r\n\xff",
    )

    exchange = RecordedExchange(
        request=request,
        response=response,
        request_timestamp=datetime(2026, 1, 27, tzinfo=UTC),
        response_timestamp=datetime(2026, 1, 27, tzinfo=UTC),
    )

    obj = exchange_to_json_obj(exchange)
    assert obj["request"]["body"] is None
    assert obj["response"]["body"] is None
    assert obj["request"]["client"] is None


def test_exchange_to_json_obj_none_body_serializes_as_null() -> None:
    request = _recorded_request(
        "GET",
        "/no-body",
        wire_raw_bytes=b"GET /no-body HTTP/1.1\r\n\r\n",
    )
    exchange = RecordedExchange(
        request=request,
        response=None,
        request_timestamp=datetime(2026, 1, 27, tzinfo=UTC),
        response_timestamp=None,
    )

    obj = exchange_to_json_obj(exchange)
    req_obj = obj["request"]
    assert isinstance(req_obj, dict)
    assert req_obj["body"] is None


def test_exchange_to_json_obj_includes_proxy_fields() -> None:
    request = _recorded_request(
        "GET",
        "http://example.com/foo?bar=1",
        body=b"",
    )
    exchange = RecordedExchange(
        request=request,
        response=None,
        request_timestamp=datetime(2026, 1, 27, tzinfo=UTC),
        response_timestamp=None,
    )

    obj = exchange_to_json_obj(exchange)
    req_obj = obj["request"]
    assert req_obj["path"] == "http://example.com/foo?bar=1"
    assert req_obj["target_host"] == "example.com"
    assert req_obj["target_port"] == 80
    assert req_obj["effective_path"] == "/foo?bar=1"


def test_jsonl_writer_writes_single_line() -> None:
    request = _recorded_request("GET", "/", body=b"")
    exchange = RecordedExchange(
        request=request,
        response=None,
        request_timestamp=datetime(2026, 1, 27, tzinfo=UTC),
        response_timestamp=None,
    )

    out = io.StringIO()
    writer = JSONLTrafficWriter(out)
    writer.write_exchange(exchange)

    lines = [line for line in out.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["request"]["raw_wire_bytes"] == ""


def test_dump_server_traffic_jsonl_supports_start_end(tmp_path) -> None:
    server = AsyncHTTPTestServer()
    exchange1 = RecordedExchange(
        request=_recorded_request("GET", "/a"),
        response=None,
        request_timestamp=datetime(2026, 1, 27, tzinfo=UTC),
        response_timestamp=None,
    )
    exchange2 = RecordedExchange(
        request=_recorded_request("GET", "/b"),
        response=None,
        request_timestamp=datetime(2026, 1, 27, tzinfo=UTC),
        response_timestamp=None,
    )
    server.exchanges.extend([exchange1, exchange2])

    output_path = tmp_path / "traffic.jsonl"
    written = dump_server_traffic_jsonl(
        server,
        output_path,
        start=1,
    )
    assert written == 1

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["request"]["path"] == "/b"


def test_dump_server_traffic_jsonl_rejects_negative_start() -> None:
    server = AsyncHTTPTestServer()
    with pytest.raises(ValueError, match="start must be non-negative"):
        dump_server_traffic_jsonl(server, io.StringIO(), start=-1)
