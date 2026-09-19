"""Close the connection without responding, then let a retry succeed."""

import asyncio

import httpx

from localstub.server import AsyncHTTPTestServer, CloseConnection, HTTPResponse


async def main() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        httpx.AsyncClient() as client,
    ):
        server.set_response_sequence([
            CloseConnection(),
            HTTPResponse.json({"status": "ready"}),
        ])

        # The server reads the first request in full, writes nothing,
        # and closes the connection, so the client sees a protocol error.
        try:
            await client.get(server.url)
        except httpx.RemoteProtocolError:
            pass
        else:
            raise AssertionError("expected the first request to fail")

        # Every connection ends with exactly one close event.
        closed = await server.next_closed_connection(timeout=1.0)
        assert closed.reason == "close_response"
        assert closed.phase == "response"
        assert closed.requests_completed == 1

        # The retry opens a new connection and gets the next response.
        response = await client.get(server.url)
        assert response.json() == {"status": "ready"}

        assert len(server.requests) == 2
        assert server.exchanges[0].response is None
        assert server.exchanges[0].closed is closed
        assert server.exchanges[1].response is not None


if __name__ == "__main__":
    asyncio.run(main())
