import asyncio
import ssl

import httpx
import pytest
import trustme

from localstub.server import AsyncHTTPTestServer
from localstub.tls_proxy import AsyncTLSInterceptProxy


@pytest.mark.asyncio
async def test_forward_to_plain_http_when_tls_disabled():
    """Forward mode: proxy relays to HTTP upstream when TLS is disabled."""

    async with AsyncHTTPTestServer() as upstream:
        upstream.set_json_response({"upstream": True})

        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            upstream_host, upstream_port = upstream.host, upstream.port
            assert upstream_host is not None
            assert upstream_port is not None

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{upstream_host}:{upstream_port}/forward",
                    follow_redirects=False,
                )

        recorded = await upstream.next_request(timeout=1.0)
        assert recorded.path == "/forward"
        assert response.status_code == 200
        assert response.json() == {"upstream": True}


@pytest.mark.asyncio
async def test_forward_uses_tls_when_verify_disabled_custom_port():
    ca = trustme.CA()
    cert = ca.issue_cert("localhost", "127.0.0.1")
    server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    cert.configure_cert(server_ctx)
    server_ctx.set_alpn_protocols(["http/1.1"])
    server_ctx.options |= ssl.OP_NO_COMPRESSION

    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        body = b"tls-ok"
        writer.write(
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Length: 6\r\n"
            b"Content-Type: text/plain\r\n"
            b"\r\n" + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(
        handle,
        "127.0.0.1",
        0,
        ssl=server_ctx,
    )
    server_host, server_port = server.sockets[0].getsockname()[:2]
    assert server_port != 443

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=True,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/tls"
                )

        assert response.status_code == 200
        assert response.text == "tls-ok"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.status == 200
        assert recorded_response.body == "tls-ok"
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_forward_preserves_close_delimited_response_bodies():
    async def handle(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        await reader.readuntil(b"\r\n\r\n")
        body = b"streamed close body"
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\n" + body
        )
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    server_host, server_port = server.sockets[0].getsockname()[:2]

    try:
        async with AsyncTLSInterceptProxy(
            server=None,
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{server_host}:{server_port}/close-body"
                )

        assert response.status_code == 200
        assert response.text == "streamed close body"

        recorded_response = await proxy.next_response(timeout=1.0)
        assert recorded_response.body == "streamed close body"
    finally:
        server.close()
        await server.wait_closed()
