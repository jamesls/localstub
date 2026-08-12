from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import socket
import ssl
import sys
from pathlib import Path

import httpx
import pytest

from localstub.cli import parse_args, run_tls_proxy
from localstub.config import load_config
from localstub.server import AsyncHTTPTestServer
from localstub.tlsproxy import AsyncTLSInterceptProxy


@pytest.mark.asyncio
async def test_cli_intercept_mode_writes_recorded_traffic(
    tmp_path: Path,
) -> None:
    with socket.socket() as available_port:
        available_port.bind(("127.0.0.1", 0))
        port = available_port.getsockname()[1]

    output_path = tmp_path / "traffic.jsonl"
    ca_dir = tmp_path / "ca"
    args = parse_args([
        "--mode",
        "intercept",
        "--port",
        str(port),
        "--output",
        str(output_path),
        "--ca-dir",
        str(ca_dir),
    ])
    cli_task = asyncio.create_task(run_tls_proxy(args))

    try:
        ca_path = ca_dir / "ca.pem"
        async with asyncio.timeout(2.0):
            while not ca_path.exists():
                await asyncio.sleep(0.01)

        verify_context = ssl.create_default_context(cafile=str(ca_path))
        async with httpx.AsyncClient(
            proxy=f"http://127.0.0.1:{port}",
            verify=verify_context,
            http2=False,
        ) as client:
            response: httpx.Response | None = None
            async with asyncio.timeout(2.0):
                while response is None:
                    try:
                        response = await client.get(
                            "https://example.com/recorded"
                        )
                    except httpx.TransportError:
                        await asyncio.sleep(0.01)

        assert response.status_code == 200
        async with asyncio.timeout(2.0):
            while (
                not output_path.exists() or not output_path.read_text().strip()
            ):
                await asyncio.sleep(0.01)

        records = output_path.read_text().splitlines()
        assert len(records) == 1
        assert json.loads(records[0])["request"]["path"] == "/recorded"
    finally:
        cli_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cli_task


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="asyncio signal handlers are unavailable on Windows",
)
@pytest.mark.asyncio
async def test_cli_writes_exchange_completed_during_shutdown(
    tmp_path: Path,
) -> None:
    with socket.socket() as available_port:
        available_port.bind(("127.0.0.1", 0))
        port = available_port.getsockname()[1]

    output_path = tmp_path / "traffic.jsonl"
    ca_dir = tmp_path / "ca"
    args = parse_args([
        "--mode",
        "intercept",
        "--port",
        str(port),
        "--output",
        str(output_path),
        "--ca-dir",
        str(ca_dir),
    ])
    cli_task = asyncio.create_task(run_tls_proxy(args))
    loop = asyncio.get_running_loop()
    writer: asyncio.StreamWriter | None = None

    try:
        ca_path = ca_dir / "ca.pem"
        async with asyncio.timeout(2.0):
            while not ca_path.exists():
                await asyncio.sleep(0.01)

        reader: asyncio.StreamReader | None = None
        async with asyncio.timeout(2.0):
            while writer is None:
                try:
                    reader, writer = await asyncio.open_connection(
                        "127.0.0.1",
                        port,
                    )
                except ConnectionRefusedError:
                    await asyncio.sleep(0.01)

        assert reader is not None
        writer.write(
            b"CONNECT example.com:443 HTTP/1.1\r\n"
            b"Host: example.com:443\r\n"
            b"\r\n"
        )
        await writer.drain()
        assert (
            await asyncio.wait_for(
                reader.readuntil(b"\r\n\r\n"),
                timeout=1.0,
            )
        ).startswith(b"HTTP/1.1 200")

        verify_context = ssl.create_default_context(cafile=str(ca_path))
        await writer.start_tls(
            verify_context,
            server_hostname="example.com",
        )
        writer.write(
            b"POST /during-shutdown HTTP/1.1\r\n"
            b"Host: example.com\r\n"
            b"Content-Length: 2\r\n"
            b"Connection: close\r\n"
            b"\r\n"
            b"a"
        )
        await writer.drain()
        await asyncio.sleep(0.05)

        os.kill(os.getpid(), signal.SIGTERM)
        await asyncio.sleep(0.2)

        writer.write(b"b")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=1.0)
        assert response.startswith(b"HTTP/1.1 200")
        await asyncio.wait_for(cli_task, timeout=2.0)

        records = output_path.read_text().splitlines()
        assert len(records) == 1
        assert json.loads(records[0])["request"]["path"] == "/during-shutdown"
    finally:
        loop.remove_signal_handler(signal.SIGTERM)
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        if not cli_task.done():
            cli_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cli_task


@pytest.mark.asyncio
async def test_intercept_mode_with_single_json_response(
    tmp_path,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({
            "response": {
                "type": "json",
                "body": {"intercepted": True},
                "status": 200,
            }
        })
    )

    config = load_config(config_file)
    server = AsyncHTTPTestServer()
    assert config.single_response is not None
    server.set_default_response(config.single_response)

    async with AsyncTLSInterceptProxy(server=server) as proxy:
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            response = await client.get("https://example.com/test")

        assert response.status_code == 200
        assert response.json() == {"intercepted": True}


@pytest.mark.asyncio
async def test_intercept_mode_with_text_response(
    tmp_path,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({
            "response": {
                "type": "text",
                "body": "Hello from intercept",
                "status": 201,
            }
        })
    )

    config = load_config(config_file)
    server = AsyncHTTPTestServer()
    assert config.single_response is not None
    server.set_default_response(config.single_response)

    async with AsyncTLSInterceptProxy(server=server) as proxy:
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            response = await client.get("https://example.com/text")

        assert response.status_code == 201
        assert response.text == "Hello from intercept"


@pytest.mark.asyncio
async def test_intercept_mode_with_response_sequence(
    tmp_path,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({
            "responses": [
                {"type": "json", "body": {"retry": 1}, "status": 503},
                {"type": "json", "body": {"retry": 2}, "status": 503},
                {"type": "json", "body": {"success": True}, "status": 200},
            ]
        })
    )

    config = load_config(config_file)
    server = AsyncHTTPTestServer()
    assert config.response_sequence is not None
    server.set_response_sequence(config.response_sequence)

    async with AsyncTLSInterceptProxy(server=server) as proxy:
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            resp1 = await client.get("https://example.com/api")
            assert resp1.status_code == 503
            assert resp1.json() == {"retry": 1}

            resp2 = await client.get("https://example.com/api")
            assert resp2.status_code == 503
            assert resp2.json() == {"retry": 2}

            resp3 = await client.get("https://example.com/api")
            assert resp3.status_code == 200
            assert resp3.json() == {"success": True}


@pytest.mark.asyncio
async def test_intercept_mode_with_custom_headers(
    tmp_path,
) -> None:
    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({
            "response": {
                "type": "json",
                "body": {},
                "status": 200,
                "headers": {
                    "X-Custom-Header": "custom-value",
                    "X-Request-Id": "test-123",
                },
            }
        })
    )

    config = load_config(config_file)
    server = AsyncHTTPTestServer()
    assert config.single_response is not None
    server.set_default_response(config.single_response)

    async with AsyncTLSInterceptProxy(server=server) as proxy:
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            response = await client.get("https://example.com/headers")

        assert response.status_code == 200
        assert response.headers["X-Custom-Header"] == "custom-value"
        assert response.headers["X-Request-Id"] == "test-123"


@pytest.mark.asyncio
async def test_intercept_mode_with_raw_base64_response(
    tmp_path,
) -> None:
    raw_data = b"binary data"
    b64_data = base64.b64encode(raw_data).decode("utf-8")

    config_file = tmp_path / "config.json"
    config_file.write_text(
        json.dumps({
            "response": {
                "type": "raw",
                "body": b64_data,
                "encoding": "base64",
                "status": 200,
                "headers": {"Content-Type": "application/octet-stream"},
            }
        })
    )

    config = load_config(config_file)
    server = AsyncHTTPTestServer()
    assert config.single_response is not None
    server.set_default_response(config.single_response)

    async with AsyncTLSInterceptProxy(server=server) as proxy:
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            response = await client.get("https://example.com/binary")

        assert response.status_code == 200
        assert response.content == raw_data
