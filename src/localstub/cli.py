from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Protocol, TextIO

import anyio
import httpx
from rich.logging import RichHandler
from rich.syntax import Syntax

from localstub.ca import TLSProxyCA
from localstub.config import load_config
from localstub.console import console
from localstub.handlers import handle_expect_header
from localstub.http.exchange import RecordedExchange
from localstub.http.utils import maybe_await
from localstub.server import AsyncHTTPTestServer
from localstub.tlsproxy import AsyncTLSInterceptProxy
from localstub.traffic_jsonl import exchange_to_json_obj

DEFAULT_PORT = 8888
type TrafficOutput = TextIO | anyio.AsyncFile[str]


def _detect_syntax(body: str, headers: str) -> str:
    """Detect syntax type from headers or body heuristics."""
    headers_lower = headers.lower()
    if "application/json" in headers_lower or "text/json" in headers_lower:
        return "json"
    if "application/xml" in headers_lower or "text/xml" in headers_lower:
        return "xml"
    body_stripped = body.strip()
    if body_stripped.startswith(("{", "[")):
        return "json"
    if body_stripped.startswith("<"):
        return "xml"
    return "text"


def _print_http_block(
    wire_bytes: bytes,
    label: str,
    color: str,
    status: int | None = None,
) -> None:
    """Print an HTTP request or response block with rich formatting."""
    text = wire_bytes.decode("utf-8", errors="replace")

    if "\r\n\r\n" in text:
        headers, body = text.split("\r\n\r\n", 1)
    else:
        headers, body = text, ""

    # Build the label with optional status
    if status is not None:
        if 200 <= status < 300:
            status_style = "green"
        elif 300 <= status < 400:
            status_style = "yellow"
        else:
            status_style = "red"
        label_text = f"[bold {color}]── {label}[/] [{status_style}]{status}[/]"
    else:
        label_text = f"[bold {color}]── {label}[/]"

    # Print with extra spacing before
    console.print()
    console.print()
    console.print(label_text)
    console.print()

    # Print headers with cleaner syntax highlighting
    theme = 'nord'
    console.print(Syntax(headers, "http", theme=theme, word_wrap=True))

    # Print body with detected syntax highlighting
    if body.strip():
        console.print()
        syntax = _detect_syntax(body, headers)
        if syntax in ("json", "xml"):
            console.print(
                Syntax(
                    body.strip(),
                    syntax,
                    theme=theme,
                    word_wrap=True,
                )
            )
        else:
            console.print(f"{body.strip()}")

    # End marker
    console.print()
    console.print(f"[{color}]──[/]")


def main() -> None:
    """Entry point for the lstub CLI."""
    args = parse_args()
    try:
        asyncio.run(run_proxy(args))
    except KeyboardInterrupt:
        pass


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog="lstub",
        description="TLS MITM proxy for inspecting HTTPS traffic",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to listen on (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Path to JSONL file for persisting traffic",
    )
    parser.add_argument(
        "--ca-dir",
        type=Path,
        help="Directory to persist/load CA certificate and key",
    )
    parser.add_argument(
        "--log-level",
        default="DEBUG",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: DEBUG)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["forward", "intercept", "http-proxy"],
        default="forward",
        help=(
            "Proxy mode: forward (TLS to upstream), intercept (TLS mock), "
            "http-proxy (HTTP forward proxy) (default: forward)"
        ),
    )
    parser.add_argument(
        "-f",
        "--config-file",
        type=Path,
        help="Path to JSON config file for intercept mode responses",
    )
    return parser.parse_args(args)


async def run_proxy(args: argparse.Namespace) -> None:
    """Start and run the proxy."""
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        handlers=[RichHandler(console=console, show_path=False)],
        format="%(message)s",
    )

    if args.mode == "http-proxy":
        await run_http_proxy(args)
    else:
        await run_tls_proxy(args)


async def run_http_proxy(args: argparse.Namespace) -> None:
    """Start and run the HTTP forward proxy."""
    # Create forwarder if not in record-only mode (config file provided)
    forwarder: httpx.AsyncClient | None = None
    mode_label: str
    if args.config_file:
        # Record mode: no forwarder, use configured responses
        mode_label = "record"
    else:
        # Forward mode: forward to upstream
        forwarder = httpx.AsyncClient(trust_env=False)
        mode_label = "forward"

    server = AsyncHTTPTestServer(
        port=args.port,
        on_headers_received=handle_expect_header,
        proxy_forwarder=forwarder,
    )

    # Configure responses if config file provided
    if args.config_file:
        config = load_config(args.config_file)
        if config.response_sequence:
            server.set_response_sequence(config.response_sequence)
        elif config.single_response:
            server.set_default_response(config.single_response)

    async with server:
        console.print()
        console.print(
            f"[bold cyan]lstub[/] HTTP proxy listening on "
            f"[bold]{server.host}:{server.port}[/]"
        )
        console.print(f"[dim]Mode:[/] {mode_label}")
        console.print("[dim]Press Ctrl+C to stop[/]")
        console.print()

        shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, shutdown_event.set)
            loop.add_signal_handler(signal.SIGTERM, shutdown_event.set)
        except (NotImplementedError, RuntimeError):
            pass

        async with AsyncExitStack() as output_stack:
            output_file: TrafficOutput | None = None
            if args.output:
                output_file = await output_stack.enter_async_context(
                    await anyio.open_file(args.output, "a", encoding="utf-8")
                )

            try:
                await process_http_proxy_traffic(
                    server, output_file, shutdown_event
                )
            finally:
                if forwarder:
                    await forwarder.aclose()


async def run_tls_proxy(args: argparse.Namespace) -> None:
    """Start and run the TLS proxy."""
    server: AsyncHTTPTestServer | None = None
    if args.mode == "intercept":
        server = AsyncHTTPTestServer()
        if args.config_file:
            config = load_config(args.config_file)
            if config.response_sequence:
                server.set_response_sequence(config.response_sequence)
            elif config.single_response:
                server.set_default_response(config.single_response)

    ca: TLSProxyCA | None = None
    if args.ca_dir:
        ca = TLSProxyCA.from_directory(args.ca_dir)

    proxy = AsyncTLSInterceptProxy(
        ca=ca,
        listen_port=args.port,
        default_mode=args.mode,
        server=server,
        verify_upstream=True,
        upstream_tls=True,
    )

    async with proxy:
        host, port = proxy.address
        console.print()
        console.print(
            f"[bold cyan]lstub[/] listening on [bold]{host}:{port}[/]"
        )
        console.print(f"[dim]CA certificate:[/] {proxy.ca.ca_pem_path()}")
        console.print(f"[dim]Keystore:[/] {proxy.ca.ca_pkcs12_path()}")
        console.print("[dim]Press Ctrl+C to stop[/]")
        console.print()

        shutdown_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGINT, shutdown_event.set)
            loop.add_signal_handler(signal.SIGTERM, shutdown_event.set)
        except (NotImplementedError, RuntimeError):
            # Signal handlers may be unsupported (e.g. on Windows)
            pass

        async with AsyncExitStack() as output_stack:
            output_file: TrafficOutput | None = None
            if args.output:
                output_file = await output_stack.enter_async_context(
                    await anyio.open_file(args.output, "a", encoding="utf-8")
                )

            if server is None:
                await process_traffic(proxy, output_file, shutdown_event)
            else:
                await process_http_proxy_traffic(
                    server, output_file, shutdown_event
                )


class _TrafficSource(Protocol):
    """Shared interface for TLS proxy and HTTP server."""

    async def next_exchange(
        self, timeout: float | None = None
    ) -> RecordedExchange: ...

    def next_exchange_nowait(self) -> RecordedExchange | None: ...

    @property
    def dropped_exchanges(self) -> int: ...


async def _emit_exchange(
    exchange: RecordedExchange,
    output_file: TrafficOutput | None,
) -> None:
    """Print an exchange to the console and persist it if configured."""
    request = exchange.request
    response = exchange.response

    _print_http_block(
        request.wire_raw_bytes or b"",
        "REQUEST",
        "green",
    )

    if response is not None:
        _print_http_block(
            response.wire_raw_bytes,
            "RESPONSE",
            "blue",
            status=response.status,
        )

    if output_file is not None:
        record = json.dumps(exchange_to_json_obj(exchange)) + "\n"
        await maybe_await(output_file.write(record))
        await maybe_await(output_file.flush())


async def _process_traffic(
    source: _TrafficSource,
    output_file: TrafficOutput | None,
    shutdown_event: asyncio.Event,
) -> None:
    """Poll *source* for completed request/response exchanges.

    Works for both the TLS intercept proxy and the HTTP forward
    proxy server.
    """
    while not shutdown_event.is_set():
        try:
            exchange = await source.next_exchange(timeout=0.1)
        except TimeoutError:
            continue
        await _emit_exchange(exchange, output_file)

    # Drain exchanges that completed before shutdown so they still
    # reach the console and JSONL output.
    while (exchange := source.next_exchange_nowait()) is not None:
        await _emit_exchange(exchange, output_file)

    if source.dropped_exchanges:
        console.print(
            f"[yellow]{source.dropped_exchanges} exchange(s) were "
            "dropped before they could be recorded[/]"
        )


async def process_traffic(
    proxy: AsyncTLSInterceptProxy,
    output_file: TrafficOutput | None,
    shutdown_event: asyncio.Event,
) -> None:
    """Poll TLS proxy for completed exchanges."""
    await _process_traffic(proxy, output_file, shutdown_event)


async def process_http_proxy_traffic(
    server: AsyncHTTPTestServer,
    output_file: TrafficOutput | None,
    shutdown_event: asyncio.Event,
) -> None:
    """Poll HTTP server for completed exchanges."""
    await _process_traffic(server, output_file, shutdown_event)
