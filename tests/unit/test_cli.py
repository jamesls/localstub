from __future__ import annotations

import asyncio
import io
import pytest
from email.message import Message
from pathlib import Path

from localstub.cli import (
    DEFAULT_PORT,
    build_record,
    parse_args,
    process_traffic,
)
from localstub.server import HTTPRequest
from localstub.tlsproxy import RecordedResponse


class TestParseArgs:
    def test_parse_args_with_no_args_returns_defaults(self) -> None:
        args = parse_args([])
        assert args.port == DEFAULT_PORT
        assert args.output is None
        assert args.ca_dir is None

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

    def test_parse_args_with_ca_dir_flag_sets_path(self) -> None:
        args = parse_args(["--ca-dir", "/tmp/my-ca"])
        assert args.ca_dir == Path("/tmp/my-ca")

    def test_parse_args_with_all_flags_sets_all_values(self) -> None:
        args = parse_args([
            "-p",
            "7777",
            "-o",
            "/tmp/out.jsonl",
            "--ca-dir",
            "/tmp/my-ca",
        ])
        assert args.port == 7777
        assert args.output == Path("/tmp/out.jsonl")
        assert args.ca_dir == Path("/tmp/my-ca")

    def test_parse_args_mode_defaults_to_forward(self) -> None:
        args = parse_args([])
        assert args.mode == "forward"

    def test_parse_args_with_short_mode_flag_sets_mode(self) -> None:
        args = parse_args(["-m", "intercept"])
        assert args.mode == "intercept"

    def test_parse_args_with_long_mode_flag_sets_mode(self) -> None:
        args = parse_args(["--mode", "intercept"])
        assert args.mode == "intercept"

    def test_parse_args_mode_forward_is_valid(self) -> None:
        args = parse_args(["--mode", "forward"])
        assert args.mode == "forward"

    def test_parse_args_config_file_defaults_to_none(self) -> None:
        args = parse_args([])
        assert args.config_file is None

    def test_parse_args_with_short_config_file_flag_sets_path(self) -> None:
        args = parse_args(["-f", "/tmp/config.json"])
        assert args.config_file == Path("/tmp/config.json")

    def test_parse_args_with_long_config_file_flag_sets_path(self) -> None:
        args = parse_args(["--config-file", "/tmp/config.json"])
        assert args.config_file == Path("/tmp/config.json")

    def test_parse_args_with_intercept_mode_and_config_file(self) -> None:
        args = parse_args(["-m", "intercept", "-f", "/tmp/stub.json"])
        assert args.mode == "intercept"
        assert args.config_file == Path("/tmp/stub.json")


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


class TestProcessTrafficNoResponse:
    @pytest.mark.asyncio
    async def test_logs_request_when_no_recorded_response(self) -> None:
        # Prepare a single recorded request that will have no corresponding
        # recorded response from the proxy.
        req_headers = Message()
        req_headers["Host"] = "example.com"
        request = HTTPRequest(
            method="GET",
            path="/no-upstream",
            headers=req_headers,
            body="",
            wire_raw_bytes=b"GET /no-upstream HTTP/1.1\r\n\r\n",
            client=("127.0.0.1", 55555),
        )

        class _NoResponseProxy:
            def __init__(self) -> None:
                self._given = False

            async def next_request(
                self, timeout: float | None = None
            ) -> HTTPRequest:  # type: ignore[override]
                if not self._given:
                    self._given = True
                    return request
                # After first request, behave like a timeout poll loop
                await asyncio.sleep(0)
                raise asyncio.TimeoutError()

            async def next_response(  # type: ignore[override]
                self, timeout: float | None = None
            ) -> RecordedResponse:
                # Never produce a response; always time out quickly
                await asyncio.sleep(0)
                raise asyncio.TimeoutError()

        proxy = _NoResponseProxy()
        out = io.StringIO()
        shutdown = asyncio.Event()

        # Run the traffic processor in the background with aggressive
        # timeouts so the test completes quickly.
        task = asyncio.create_task(
            process_traffic(
                proxy,  # type: ignore[arg-type]
                out,
                shutdown,
                response_timeout=0.01,
                response_max_timeouts=2,
            )
        )

        # Give the loop a moment to process, then stop it.
        await asyncio.sleep(0.05)
        shutdown.set()
        await asyncio.wait_for(task, timeout=1.0)

        # Exactly one JSONL record should be written with response set to null
        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        assert len(lines) == 1

        record = __import__("json").loads(lines[0])
        assert record["request"]["path"] == "/no-upstream"
        assert record["response"] is None
