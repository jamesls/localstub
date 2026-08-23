"""Return transient failures followed by a successful response."""

import asyncio

import httpx

from localstub.server import AsyncHTTPTestServer, HTTPResponse


async def main() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        httpx.AsyncClient() as client,
    ):
        server.set_response_sequence([
            HTTPResponse(status=503),
            HTTPResponse(status=503),
            HTTPResponse.json({"status": "ready"}),
        ])

        responses = [await client.get(server.url) for _ in range(3)]

        assert [response.status_code for response in responses] == [
            503,
            503,
            200,
        ]
        assert responses[-1].json() == {"status": "ready"}
        assert len(server.requests) == 3


if __name__ == "__main__":
    asyncio.run(main())
