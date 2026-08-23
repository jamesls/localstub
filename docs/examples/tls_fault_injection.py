"""Mutate a forwarded HTTPS response to test client error handling."""

import asyncio
import ssl

import httpx

from localstub.server import AsyncHTTPTestServer, ByteFlip
from localstub.tlsproxy import AsyncTLSInterceptProxy, fault_step_transformer


async def main() -> None:
    async with AsyncHTTPTestServer() as upstream:
        upstream.set_text_response("ready")

        async with AsyncTLSInterceptProxy(
            default_mode="forward",
            verify_upstream=False,
            upstream_tls=False,
            response_transformer=fault_step_transformer(
                ByteFlip(offset=0, mask=0x20)
            ),
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
                    f"https://{upstream.host}:{upstream.port}/status"
                )

            recorded = await proxy.next_response(timeout=1.0)

        assert response.text == "Ready"
        assert recorded.body == b"ready"


if __name__ == "__main__":
    asyncio.run(main())
