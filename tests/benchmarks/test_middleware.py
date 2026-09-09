from __future__ import annotations

import asyncio

from pytest_codspeed import BenchmarkFixture

from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.http.responsespec import HTTPResponse
from localstub.middleware import (
    ConnectionMeta,
    ResponderContext,
    ResponseSpec,
    ServerServices,
    compose_responder,
)
from localstub.router import HandlerMiddleware, Router, RouterMiddleware

STATIC_RESPONSE = HTTPResponse.json({"ok": True})


def _build_context() -> ResponderContext:
    request = HTTPRequest(
        method="GET",
        target="/v1/items?page=2",
        headers=Headers.from_items([
            ("Host", "localhost:8080"),
            ("Accept", "application/json"),
            ("Authorization", "Bearer 0123456789abcdef"),
        ]),
    )
    recorded = RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=b"GET /v1/items?page=2 HTTP/1.1\r\n\r\n",
        http_version="1.1",
        client=("127.0.0.1", 54321),
    )
    return ResponderContext(
        request=recorded,
        connection=ConnectionMeta(client=recorded.client),
        services=ServerServices(),
    )


def _static_response(_: ResponderContext) -> ResponseSpec:
    return STATIC_RESPONSE


# The same graph AsyncHTTPTestServer builds when no user middleware or
# builtin is configured: the router (no routes) and handler (no handler)
# middlewares both fall through to the default response.
CONTEXT = _build_context()
RESPONDER = compose_responder(
    [RouterMiddleware(Router()), HandlerMiddleware(lambda: None)],
    _static_response,
)


def test_responder_pipeline_static_json(
    benchmark: BenchmarkFixture,
    loop: asyncio.AbstractEventLoop,
) -> None:
    def dispatch_once() -> ResponseSpec:
        return loop.run_until_complete(RESPONDER(CONTEXT))

    response = benchmark(dispatch_once)

    assert response is STATIC_RESPONSE
