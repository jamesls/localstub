import httpx
import ssl
import pytest

from localstub.server import AsyncHTTPTestServer

# NOTE: AsyncTLSInterceptProxy is not implemented yet; this test defines
# the expected happy-path behavior for intercepting HTTPS requests via
# an HTTP CONNECT proxy that terminates TLS and routes into localstub.
from localstub.tls_proxy import AsyncTLSInterceptProxy


@pytest.mark.asyncio
async def test_tls_proxy_intercepts_https_request_to_localstub():
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"ok": True})

        async with AsyncTLSInterceptProxy(server=server) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_path = proxy.ca.ca_pem_path()
            verify_ctx = ssl.create_default_context(cafile=str(verify_path))

            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get("https://example.com/")

            assert response.status_code == 200
            assert response.json() == {"ok": True}

        # The intercepted request should be recorded by the underlying server
        request = await server.next_request(timeout=1.0)
        assert request.path == "/"
        assert request.headers is not None
        assert request.headers["host"] == "example.com"
