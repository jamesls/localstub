"""Inspect the exact wire bytes of a chunked request body."""

import asyncio
from collections.abc import AsyncIterator

import httpx

from localstub.server import AsyncHTTPTestServer


async def stream_body() -> AsyncIterator[bytes]:
    yield b"hello"
    yield b" "
    yield b"world"


async def main() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        httpx.AsyncClient() as client,
    ):
        server.set_json_response({"status": "received"})

        # Streaming a body makes httpx send it with chunked framing.
        response = await client.post(
            f"{server.url}upload",
            content=stream_body(),
        )

        assert response.status_code == 200
        request = server.last_request
        assert request is not None
        assert request.headers["transfer-encoding"] == "chunked"

        # The decoded payload is available as ``body``...
        assert request.body == b"hello world"

        # ...while the wire bytes keep the chunk-size lines and the
        # terminating zero-length chunk exactly as the client sent them.
        assert request.wire_body_bytes == (
            b"5\r\nhello\r\n1\r\n \r\n5\r\nworld\r\n0\r\n\r\n"
        )
        assert request.wire_raw_bytes.startswith(b"POST /upload HTTP/1.1\r\n")


if __name__ == "__main__":
    asyncio.run(main())
