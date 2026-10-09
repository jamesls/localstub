"""TCP-level control over asyncio stream connections."""

from __future__ import annotations

import asyncio
import logging
import socket
import struct
import sys

LOG = logging.getLogger(__name__)


def pack_linger_option(*, platform: str = sys.platform) -> bytes:
    """Pack the ``SO_LINGER`` value that turns close into a TCP reset.

    The option is a C struct with no Python constant: Linux and macOS
    define it as two ints, Windows as two unsigned shorts, and the
    kernel rejects the wrong size.
    """
    layout = "HH" if platform == "win32" else "ii"
    return struct.pack(layout, 1, 0)


def reset_stream(writer: asyncio.StreamWriter) -> None:
    """Request an abortive TCP close, even after writes have drained."""
    sock = writer.get_extra_info("socket")
    if sock is not None:
        try:
            sock.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, pack_linger_option()
            )
        except OSError:
            LOG.warning(
                "Could not request a TCP reset; the client may observe "
                "EOF instead",
                exc_info=True,
            )
    writer.transport.abort()
