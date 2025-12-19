"""Integration tests for CLI intercept mode."""

from __future__ import annotations

import base64
import json
import ssl

import httpx
import pytest

from localstub.server import AsyncHTTPTestServer
from localstub.tlsproxy import AsyncTLSInterceptProxy
from localstub.config import load_config


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
