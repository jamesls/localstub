"""Opening TCP and TLS connections to upstream servers."""

from __future__ import annotations

import asyncio
import ssl

import truststore


def upstream_ssl_context(verify: bool) -> ssl.SSLContext:
    """Build the TLS context used for upstream connections.

    Verified contexts use system trust (truststore); unverified
    contexts disable certificate and hostname checks entirely.
    """
    if verify:
        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ssl_ctx.check_hostname = False
    ssl_ctx.verify_mode = ssl.CERT_NONE
    return ssl_ctx


async def open_upstream_connection(
    host: str,
    port: int,
    *,
    use_tls: bool,
    verify: bool = True,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a TCP (optionally TLS) stream connection to *host*:*port*.

    Raises OSError (including ssl.SSLError) on connection failure.
    """
    ssl_ctx = upstream_ssl_context(verify) if use_tls else None
    server_hostname = host if use_tls else None
    return await asyncio.open_connection(
        host, port, ssl=ssl_ctx, server_hostname=server_hostname
    )
