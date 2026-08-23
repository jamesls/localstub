"""Forward HTTPS client traffic and record the exchange."""

import asyncio
import ssl

import httpx

from localstub.server import AsyncHTTPTestServer
from localstub.tlsproxy import AsyncTLSInterceptProxy


async def main() -> None:
    async with AsyncHTTPTestServer() as upstream:
        upstream.set_text_response("upstream response")

        async with AsyncTLSInterceptProxy(
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
        ) as proxy:
            verify = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )
            async with httpx.AsyncClient(
                proxy=proxy.endpoint_url,
                verify=verify,
                http2=False,
            ) as client:
                response = await client.get(
                    f"https://{upstream.host}:{upstream.port}/health"
                )

            exchange = await proxy.next_exchange(timeout=1.0)

        assert response.text == "upstream response"
        assert exchange.request.target == "/health"
        assert exchange.response is not None
        assert exchange.response.status == 200
        assert exchange.response.body == b"upstream response"


if __name__ == "__main__":
    asyncio.run(main())
