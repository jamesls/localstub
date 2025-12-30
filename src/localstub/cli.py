from __future__ import annotations

import argparse
import asyncio
import json
import logging
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

from rich.logging import RichHandler
from rich.syntax import Syntax

import httpx

from localstub.ca import TLSProxyCA
from localstub.config import load_config
from localstub.console import console
from localstub.http.request import HTTPRequest
from localstub.server import AsyncHTTPTestServer
from localstub.tlsproxy import AsyncTLSInterceptProxy, RecordedResponse

DEFAULT_PORT = 8888


def _detect_syntax(body: str, headers: str) -> str:
    """Detect syntax type from headers or body heuristics."""
    headers_lower = headers.lower()
    if "application/json" in headers_lower or "text/json" in headers_lower:
        return "json"
    if "application/xml" in headers_lower or "text/xml" in headers_lower:
        return "xml"
    body_stripped = body.strip()
    if body_stripped.startswith("{") or body_stripped.startswith("["):
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
        forwarder = httpx.AsyncClient()
        mode_label = "forward"

    server = AsyncHTTPTestServer(
        port=args.port,
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

        output_file: TextIO | None = None
        if args.output:
            output_file = open(args.output, "a")

        try:
            await process_http_proxy_traffic(
                server, output_file, shutdown_event
            )
        finally:
            if output_file:
                output_file.close()
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

        output_file: TextIO | None = None
        if args.output:
            output_file = open(args.output, "a")

        try:
            await process_traffic(proxy, output_file, shutdown_event)
        finally:
            if output_file:
                output_file.close()


async def process_traffic(
    proxy: AsyncTLSInterceptProxy,
    output_file: TextIO | None,
    shutdown_event: asyncio.Event,
    *,
    response_timeout: float = 5.0,
    response_max_timeouts: int = 3,
) -> None:
    """Poll for recorded requests/responses and output them.

    If the TLS proxy cannot obtain an upstream response it will not enqueue
    a ``RecordedResponse``. In that case we treat repeated timeouts while
    awaiting ``next_response()`` as "no response" and log the request with
    ``response=None`` so traffic recording continues.
    """
    while not shutdown_event.is_set():
        try:
            request = await proxy.next_request(timeout=0.1)
        except asyncio.TimeoutError:
            continue

        _print_http_block(
            request.wire_raw_bytes or b"",
            "REQUEST",
            "green",
        )

        response: RecordedResponse | None = None
        timeouts = 0
        while not shutdown_event.is_set() and response is None:
            try:
                response = await proxy.next_response(timeout=response_timeout)
            except asyncio.TimeoutError:
                timeouts += 1
                if timeouts >= response_max_timeouts:
                    console.print(
                        "[yellow]No upstream response recorded; "
                        "logging without response[/]"
                    )
                    break
                continue

        if shutdown_event.is_set() and response is None:
            break

        if response:
            _print_http_block(
                response.wire_raw_bytes,
                "RESPONSE",
                "blue",
                status=response.status,
            )

        if output_file:
            record = build_record(request, response)
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()


async def process_http_proxy_traffic(
    server: AsyncHTTPTestServer,
    output_file: TextIO | None,
    shutdown_event: asyncio.Event,
) -> None:
    """Poll for recorded requests and output them (HTTP forward proxy mode)."""
    while not shutdown_event.is_set():
        try:
            request = await server.next_request(timeout=0.1)
        except asyncio.TimeoutError:
            continue

        _print_http_block(
            request.wire_raw_bytes or b"",
            "REQUEST",
            "green",
        )

        if output_file:
            record = build_http_proxy_record(request)
            output_file.write(json.dumps(record) + "\n")
            output_file.flush()


def build_http_proxy_record(request: HTTPRequest) -> dict[str, object]:
    """Build a JSON-serializable record from an HTTP proxy request."""
    request_dict: dict[str, object] = {
        "method": request.method,
        "path": request.path,
        "body": request.body,
    }
    if request.headers:
        request_dict["headers"] = dict(request.headers.items())

    # Add proxy-specific fields
    if request.is_proxy_request and request.target_uri:
        request_dict["target_host"] = request.target_uri.host
        request_dict["target_port"] = request.target_uri.port
        request_dict["effective_path"] = request.effective_path

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": request_dict,
    }


def build_record(
    request: HTTPRequest,
    response: RecordedResponse | None,
) -> dict[str, object]:
    """Build a JSON-serializable record from request/response."""
    request_dict: dict[str, object] = {
        "method": request.method,
        "path": request.path,
        "body": request.body,
    }
    if request.headers:
        request_dict["headers"] = dict(request.headers.items())

    record: dict[str, object] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request": request_dict,
    }

    if response:
        response_dict: dict[str, object] = {
            "status": response.status,
            "reason": response.reason,
            "body": response.body,
        }
        if response.headers:
            response_dict["headers"] = dict(response.headers.items())
        record["response"] = response_dict
    else:
        record["response"] = None

    return record
