"""Return a fixed response and inspect the request sent by a client."""

import asyncio

import httpx

from localstub.server import AsyncHTTPTestServer


async def main() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        httpx.AsyncClient() as client,
    ):
        server.set_json_response({"id": 123, "name": "example"})

        response = await client.post(
            f"{server.url}widgets",
            json={"color": "blue"},
        )

        assert response.json() == {"id": 123, "name": "example"}
        assert server.last_request is not None
        assert server.last_request.target == "/widgets"
        assert server.last_request.json_body == {"color": "blue"}


if __name__ == "__main__":
    asyncio.run(main())
