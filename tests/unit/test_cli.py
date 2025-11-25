from __future__ import annotations

from email.message import Message
from pathlib import Path

from localstub.cli import (
    DEFAULT_PORT,
    build_record,
    parse_args,
)
from localstub.server import HTTPRequest
from localstub.tls_proxy import RecordedResponse


class TestParseArgs:
    def test_parse_args_with_no_args_returns_defaults(self) -> None:
        args = parse_args([])
        assert args.port == DEFAULT_PORT
        assert args.output is None
        assert args.ca_cert is None

    def test_parse_args_with_short_port_flag_sets_port(self) -> None:
        args = parse_args(["-p", "9999"])
        assert args.port == 9999

    def test_parse_args_with_long_port_flag_sets_port(self) -> None:
        args = parse_args(["--port", "9999"])
        assert args.port == 9999

    def test_parse_args_with_short_output_flag_sets_path(self) -> None:
        args = parse_args(["-o", "/tmp/traffic.jsonl"])
        assert args.output == Path("/tmp/traffic.jsonl")

    def test_parse_args_with_long_output_flag_sets_path(self) -> None:
        args = parse_args(["--output", "/tmp/traffic.jsonl"])
        assert args.output == Path("/tmp/traffic.jsonl")

    def test_parse_args_with_short_ca_cert_flag_sets_path(self) -> None:
        args = parse_args(["-c", "/tmp/ca.pem"])
        assert args.ca_cert == Path("/tmp/ca.pem")

    def test_parse_args_with_long_ca_cert_flag_sets_path(self) -> None:
        args = parse_args(["--ca-cert", "/tmp/ca.pem"])
        assert args.ca_cert == Path("/tmp/ca.pem")

    def test_parse_args_with_all_flags_sets_all_values(self) -> None:
        args = parse_args([
            "-p",
            "7777",
            "-o",
            "/tmp/out.jsonl",
            "-c",
            "/tmp/ca.pem",
        ])
        assert args.port == 7777
        assert args.output == Path("/tmp/out.jsonl")
        assert args.ca_cert == Path("/tmp/ca.pem")


class TestBuildRecord:
    def test_build_record_with_request_only_sets_response_to_none(
        self,
    ) -> None:
        headers = Message()
        headers["Host"] = "example.com"
        request = HTTPRequest(
            method="GET",
            path="/test",
            headers=headers,
            body="",
            wire_raw_bytes=b"GET /test HTTP/1.1\r\n\r\n",
            client=("127.0.0.1", 12345),
        )
        record = build_record(request, None)

        assert "timestamp" in record
        assert record["request"]["method"] == "GET"
        assert record["request"]["path"] == "/test"
        assert record["request"]["headers"] == {"Host": "example.com"}
        assert record["response"] is None

    def test_build_record_with_request_and_response_includes_both(
        self,
    ) -> None:
        req_headers = Message()
        req_headers["Host"] = "example.com"
        request = HTTPRequest(
            method="POST",
            path="/api",
            headers=req_headers,
            body='{"key": "value"}',
            wire_raw_bytes=b"POST /api HTTP/1.1\r\n\r\n",
            client=("127.0.0.1", 12345),
        )

        resp_headers = Message()
        resp_headers["Content-Type"] = "application/json"
        response = RecordedResponse(
            status=200,
            reason="OK",
            headers=resp_headers,
            body='{"result": "success"}',
            wire_raw_bytes=b"HTTP/1.1 200 OK\r\n\r\n",
        )

        record = build_record(request, response)

        assert "timestamp" in record
        assert record["request"]["method"] == "POST"
        assert record["request"]["path"] == "/api"
        assert record["request"]["body"] == '{"key": "value"}'
        assert record["response"]["status"] == 200
        assert record["response"]["reason"] == "OK"
        assert record["response"]["body"] == '{"result": "success"}'
        assert record["response"]["headers"] == {
            "Content-Type": "application/json"
        }

    def test_build_record_with_none_headers_omits_headers_key(self) -> None:
        request = HTTPRequest(
            method="GET",
            path="/",
            headers=None,
            body="",
            wire_raw_bytes=b"GET / HTTP/1.1\r\n\r\n",
            client=("127.0.0.1", 12345),
        )
        record = build_record(request, None)

        assert record["request"]["method"] == "GET"
        assert "headers" not in record["request"]

    def test_build_record_with_response_none_headers_omits_headers_key(
        self,
    ) -> None:
        req_headers = Message()
        request = HTTPRequest(
            method="GET",
            path="/",
            headers=req_headers,
            body="",
            wire_raw_bytes=b"GET / HTTP/1.1\r\n\r\n",
            client=("127.0.0.1", 12345),
        )
        response = RecordedResponse(
            status=204,
            reason="No Content",
            headers=None,
            body=None,
            wire_raw_bytes=b"HTTP/1.1 204 No Content\r\n\r\n",
        )

        record = build_record(request, response)

        assert record["response"]["status"] == 204
        assert "headers" not in record["response"]
