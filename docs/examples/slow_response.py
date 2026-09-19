"""Trickle a response body to test a client's read timeout."""

import asyncio

import httpx

from localstub.server import AsyncHTTPTestServer, ThrottledTransmission


async def main() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        httpx.AsyncClient(timeout=httpx.Timeout(5.0, read=0.1)) as client,
    ):
        server.set_raw_response(b"x" * 4096)

        # Send the body 1 KiB at a time with a pause between chunks.  The
        # headers and the first chunk arrive right away; the client then
        # waits longer than its read timeout for the second chunk.
        server.set_transmission_strategy(
            ThrottledTransmission(chunk_size=1024, delay=0.5)
        )

        try:
            await client.get(f"{server.url}download")
        except httpx.ReadTimeout:
            pass
        else:
            raise AssertionError("expected the download to time out")

        assert server.last_request is not None
        assert server.last_request.target == "/download"


if __name__ == "__main__":
    asyncio.run(main())
