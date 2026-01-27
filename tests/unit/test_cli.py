from __future__ import annotations

import base64
import asyncio
import io
import json
from datetime import datetime, timezone
from email.message import Message
from pathlib import Path

import httpx
import pytest

from localstub.cli import (
    DEFAULT_PORT,
    parse_args,
    process_http_proxy_traffic,
    process_traffic,
)
from localstub.http.request import HTTPRequest
from localstub.server import AsyncHTTPTestServer, HTTPResponse
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


class TestProcessTrafficNoResponse:
    @pytest.mark.asyncio
    async def test_logs_request_when_no_recorded_response(self) -> None:
        # Prepare a single recorded request that will have no corresponding
        # recorded response from the proxy.
        headers = Message()
        headers["Host"] = "example.com"
        request = HTTPRequest(
            method="GET",
            path="/no-upstream",
            headers=headers,
            body="",
            wire_raw_bytes=b"GET /no-upstream HTTP/1.1\r\n\r\n",
            client=("127.0.0.1", 55555),
        )

        class _NoResponseProxy:
            def __init__(self) -> None:
                self._given = False

            async def next_request(
                self, timeout: float | None = None
            ) -> HTTPRequest:
                if not self._given:
                    self._given = True
                    return request
                # After first request, behave like a timeout poll loop
                await asyncio.sleep(0)
                raise asyncio.TimeoutError()

            async def next_response(
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
                proxy,
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

        record = json.loads(lines[0])
        assert record["request"]["path"] == "/no-upstream"
        assert record["response_timestamp"] is None
        assert record["response"] is None
        assert record["request"]["headers"] == {"Host": "example.com"}
        assert record["request"]["client"] == {
            "host": "127.0.0.1",
            "port": 55555,
        }
        assert (
            base64.b64decode(record["request"]["raw_wire_bytes"])
            == b"GET /no-upstream HTTP/1.1\r\n\r\n"
        )


class TestProcessHttpProxyTrafficNoResponse:
    @pytest.mark.asyncio
    async def test_uses_recorded_timestamp_when_response_missing(
        self,
    ) -> None:
        fixed_timestamp = datetime(2000, 1, 1, tzinfo=timezone.utc)

        class _FixedTimestampProvider:
            def __init__(self, ts: datetime) -> None:
                self._ts = ts

            def now(self) -> datetime:
                return self._ts

        def _boom_handler(_: HTTPRequest) -> HTTPResponse:
            raise RuntimeError("boom")

        out = io.StringIO()
        shutdown = asyncio.Event()

        async with AsyncHTTPTestServer(
            timestamp_provider=_FixedTimestampProvider(fixed_timestamp),
        ) as server:
            server.handler = _boom_handler

            task = asyncio.create_task(
                process_http_proxy_traffic(
                    server,
                    out,
                    shutdown,
                    response_timeout=0.01,
                    response_max_timeouts=2,
                )
            )

            async with httpx.AsyncClient(timeout=0.2) as client:
                with pytest.raises(httpx.HTTPError):
                    await client.get(f"{server.url}boom")

            await asyncio.sleep(0.05)
            shutdown.set()
            await asyncio.wait_for(task, timeout=1.0)

        lines = [line for line in out.getvalue().splitlines() if line.strip()]
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["request"]["path"] == "/boom"
        assert record["timestamp"] == fixed_timestamp.isoformat()
        assert record["response_timestamp"] is None
        assert record["response"] is None
