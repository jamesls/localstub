from __future__ import annotations

import asyncio
import socket
from typing import Any
from unittest.mock import Mock, create_autospec


class RecordingSleep:
    """An injectable sleep that records each delay and returns at once."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)


def fake_writer(
    *,
    peer: tuple[str, int] | None = None,
    sock: socket.socket | None = None,
) -> Mock:
    """An autospec ``StreamWriter`` that closing or aborting marks closing.

    ``peer`` and ``sock`` are reported as the ``peername`` and
    ``socket`` transport extras.
    """
    writer = create_autospec(asyncio.StreamWriter, instance=True)
    writer.is_closing.return_value = False
    extras: dict[str, Any] = {"peername": peer, "socket": sock}

    def get_extra_info(name: str, default: Any = None) -> Any:
        return extras.get(name, default)

    def close() -> None:
        writer.is_closing.return_value = True

    writer.get_extra_info.side_effect = get_extra_info
    writer.close.side_effect = close
    writer.transport = create_autospec(asyncio.Transport, instance=True)
    writer.transport.abort.side_effect = close
    return writer


async def settle() -> None:
    """Let pending callbacks and tasks run for a few loop turns."""
    for _ in range(5):
        await asyncio.sleep(0)
