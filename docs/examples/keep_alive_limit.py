"""Close keep-alive connections after a fixed number of requests."""

import asyncio

import httpx

from localstub.server import AsyncHTTPTestServer


async def main() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        httpx.AsyncClient() as client,
    ):
        server.set_json_response({"status": "ok"})

        # Allow two requests per connection.  The second response carries
        # ``Connection: close`` and the server closes after sending it.
        server.set_keep_alive(max_requests=2)

        first = await client.get(server.url)
        second = await client.get(server.url)
        closed = await server.next_closed_connection(timeout=1.0)

        # The client's pool must notice the close and open a fresh
        # connection for the next request.
        third = await client.get(server.url)

        assert "connection" not in first.headers
        assert second.headers["connection"] == "close"
        assert third.status_code == 200

        assert closed.reason == "max_requests"
        assert closed.requests_completed == 2

        # Each recorded request carries the client (host, port) address,
        # so the connection reuse pattern is visible after the fact.
        clients = [request.client for request in server.requests]
        assert clients[0] == clients[1]
        assert clients[1] != clients[2]


if __name__ == "__main__":
    asyncio.run(main())
