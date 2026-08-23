"""Intercept an HTTPS request and return a local stub response."""

import asyncio
import ssl

import httpx

from localstub.server import AsyncHTTPTestServer
from localstub.tlsproxy import AsyncTLSInterceptProxy


async def main() -> None:
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"source": "local stub"})

        async with AsyncTLSInterceptProxy(server=server) as proxy:
            verify = ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            )
            async with httpx.AsyncClient(
                proxy=proxy.endpoint_url,
                verify=verify,
                http2=False,
            ) as client:
                response = await client.get("https://api.example.test/widgets")

        assert response.json() == {"source": "local stub"}
        assert server.last_request is not None
        assert server.last_request.target == "/widgets"
        assert server.last_request.headers["host"] == "api.example.test"


if __name__ == "__main__":
    asyncio.run(main())
