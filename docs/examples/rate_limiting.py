"""Throttle requests and honor the Retry-After header."""

import asyncio

import httpx

from localstub.server import AsyncHTTPTestServer


async def main() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        httpx.AsyncClient() as client,
    ):
        server.set_json_response({"status": "ok"})

        # A token bucket refilling at one request per second.  Requests
        # that arrive with no token available get 429 Too Many Requests.
        server.set_throttle(rate_per_second=1.0)

        first = await client.get(server.url)
        assert first.status_code == 200

        throttled = await client.get(server.url)
        assert throttled.status_code == 429
        retry_after = int(throttled.headers["retry-after"])
        assert retry_after == 1

        # Waiting for the advertised interval refills the bucket.
        await asyncio.sleep(retry_after)
        retried = await client.get(server.url)
        assert retried.status_code == 200

        # Throttled requests are still recorded.
        assert len(server.requests) == 3


if __name__ == "__main__":
    asyncio.run(main())
