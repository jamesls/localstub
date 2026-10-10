from __future__ import annotations

import asyncio


def _content_length(head: bytes) -> int:
    for line in head.split(b"\r\n"):
        name, sep, value = line.partition(b":")
        if sep and name.strip().lower() == b"content-length":
            return int(value.strip())
    return 0


async def read_http_response(
    reader: asyncio.StreamReader,
    *,
    timeout: float = 1.0,
) -> bytes:
    """Read one Content-Length framed response, head and body."""
    head = await asyncio.wait_for(
        reader.readuntil(b"\r\n\r\n"), timeout=timeout
    )
    body = await asyncio.wait_for(
        reader.readexactly(_content_length(head)), timeout=timeout
    )
    return head + body
