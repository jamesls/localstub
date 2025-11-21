import ssl

import httpx
import pytest

from localstub.server import AsyncHTTPTestServer
from localstub.tls_proxy import AsyncTLSInterceptProxy


@pytest.mark.asyncio
async def test_tls_proxy_forwards_https_request_to_upstream():
    """Forward mode: proxy relays to real upstream without recording."""

    # Upstream server that responds with known JSON
    async with AsyncHTTPTestServer() as upstream:
        upstream.set_json_response({"upstream": True})

        async with AsyncTLSInterceptProxy(
            server=None,  # forward-only for this test
            default_mode="forward",  # will be added in implementation
            verify_upstream=False,  # skip verification for test upstream
        ) as proxy:
            proxy_host, proxy_port = proxy.address
            verify_ctx = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )

            # Map the CONNECT target host to the upstream server
            upstream_host, upstream_port = upstream.host, upstream.port
            assert upstream_host is not None
            assert upstream_port is not None

            # Client goes through proxy to an HTTPS URL using upstream_host
            async with httpx.AsyncClient(
                proxy=f"http://{proxy_host}:{proxy_port}",
                verify=verify_ctx,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{upstream_host}:{upstream_port}/forward",
                    follow_redirects=False,
                )

        # Since this is forward mode without recording, the proxy should not
        # have pushed anything into upstream.requests; but the upstream server
        # itself should have seen the request normally.
        recorded = await upstream.next_request(timeout=1.0)
        assert recorded.path == "/forward"
        assert response.status_code == 200
        assert response.json() == {"upstream": True}
