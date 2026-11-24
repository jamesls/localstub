"""Reject, drop, and accept WebSocket handshakes to test reconnects."""

import asyncio
import json

import websockets

from localstub import (
    AcceptWebSocket,
    AsyncHTTPTestServer,
    CloseConnection,
    HTTPResponse,
)


async def subscribe(url: str) -> list[str]:
    """A client that reconnects and resumes after the last event it saw."""
    seen: list[str] = []
    last = 0
    while True:
        try:
            async with websockets.connect(url, open_timeout=1) as ws:
                if last:
                    await ws.send(json.dumps({"resume_after": last}))
                async for raw in ws:
                    seen.append(str(raw))
                    last = json.loads(str(raw))["seq"]
                return seen
        except (
            websockets.InvalidStatus,
            websockets.InvalidMessage,
            websockets.ConnectionClosedError,
        ):
            await asyncio.sleep(0.01)


async def main() -> None:
    async with AsyncHTTPTestServer() as server:
        server.set_response_sequence([
            HTTPResponse(status=503),
            CloseConnection(reset=True),
            AcceptWebSocket(),
            AcceptWebSocket(),
        ])
        task = asyncio.create_task(subscribe(server.ws_url))

        first = await server.next_websocket(timeout=2.0)
        await first.send_text('{"seq": 1}')
        await first.drop()  # close TCP with no Close frame

        second = await server.next_websocket(timeout=2.0)
        resume = await second.receive(timeout=2.0)
        await second.send_text('{"seq": 2}')
        await second.close()

        closed = await first.wait_closed(timeout=1.0)
        assert resume.text == '{"resume_after": 1}'
        assert closed.reason == "websocket_drop"
        assert await task == ['{"seq": 1}', '{"seq": 2}']
        assert len(server.requests) == 4


if __name__ == "__main__":
    asyncio.run(main())
