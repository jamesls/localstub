"""Accept large uploads without keeping their bodies in memory."""

import asyncio

import httpx

from localstub.server import AsyncHTTPTestServer

SIZE = 64 * 1024 * 1024


async def main() -> None:
    async with (
        AsyncHTTPTestServer(
            # Keep only the most recent request and no per-connection
            # byte log, and keep only the first KiB of each body.
            recording_buffer_size=1,
            max_connection_bytes=0,
            max_recorded_body_bytes=1024,
        ) as server,
        httpx.AsyncClient() as client,
    ):
        server.set_json_response({"status": "stored"})
        body = bytes(range(256)) * (SIZE // 256)

        for _ in range(4):
            response = await client.put(
                f"{server.url}bucket/key", content=body
            )
            assert response.status_code == 200

        # Every body was read to its end, but the recording kept only
        # its first KiB and the size that was received.
        request = server.last_request
        assert request is not None
        assert request.body == body[:1024]
        assert request.full_body_size == SIZE


if __name__ == "__main__":
    asyncio.run(main())
