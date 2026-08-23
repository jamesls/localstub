"""Build a response from the incoming request."""

import asyncio

import httpx

from localstub.middleware import ResponderContext
from localstub.server import AsyncHTTPTestServer, HTTPResponse


def create_widget(ctx: ResponderContext) -> HTTPResponse:
    request_data = ctx.request.json_body
    return HTTPResponse.json(
        {"id": 123, "name": request_data["name"]},
        status=201,
    )


async def main() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        httpx.AsyncClient() as client,
    ):
        server.add_route("POST", "/widgets", create_widget)

        response = await client.post(
            f"{server.url}widgets",
            json={"name": "example"},
        )

        assert response.status_code == 201
        assert response.json() == {"id": 123, "name": "example"}


if __name__ == "__main__":
    asyncio.run(main())
