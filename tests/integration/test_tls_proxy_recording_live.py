import ssl

import httpx
import pytest

from localstub.tls_proxy import AsyncTLSInterceptProxy


@pytest.mark.asyncio
async def test_proxy_records_request_and_response_when_forwarding_to_real():
    """
    Proxy should record the decrypted HTTP request and the upstream HTTP
    response while forwarding to a real HTTPS origin.
    """

    async with AsyncTLSInterceptProxy(default_mode="forward") as proxy:
        proxy_host, proxy_port = proxy.address
        verify_ctx = ssl.create_default_context(
            cafile=str(proxy.ca.ca_pem_path())
        )

        async with httpx.AsyncClient(
            proxy=f"http://{proxy_host}:{proxy_port}",
            verify=verify_ctx,
            http2=False,
        ) as client:
            response = await client.get("https://example.com/")

    assert response.status_code == 200
    assert "Example Domain" in response.text

    recorded_request = await proxy.next_request(timeout=2.0)  # type: ignore[attr-defined]
    assert recorded_request.path == "/"
    assert recorded_request.headers is not None
    assert recorded_request.headers["host"] == "example.com"

    recorded_response = await proxy.next_response(timeout=2.0)  # type: ignore[attr-defined]
    assert recorded_response.status == 200
    assert "Example Domain" in recorded_response.body
