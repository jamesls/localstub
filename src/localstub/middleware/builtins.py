from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import httpx

from localstub.forward import Forwarder
from localstub.http.proxy import build_origin_form_request, forward_via_httpx
from localstub.http.request import (
    HTTPRequest,
    parsed_body_bytes_from_wire_raw_bytes,
)
from localstub.http.responsespec import HTTPResponse
from localstub.http.utils import maybe_await
from localstub.middleware.core import (
    ForwardProxyResponse,
    ResponderContext,
    ResponderNext,
    ResponseSpec,
)
from localstub.router import ResponderHandler, Router
from localstub.throttle import RequestThrottler, ThrottleDecision

ThrottleResponseFunc = Callable[[HTTPRequest, ThrottleDecision], HTTPResponse]
BODY_FRAMING_HEADERS = ("Content-Length", "Transfer-Encoding")


def _wire_header_values(wire_raw_bytes: bytes, name: str) -> list[str]:
    header_end = wire_raw_bytes.find(b"\r\n\r\n")
    if header_end == -1:
        return []

    name_bytes = name.lower().encode("ascii")
    values: list[str] = []
    header_block = wire_raw_bytes[:header_end]
    for line in header_block.split(b"\r\n")[1:]:
        header_name, separator, header_value = line.partition(b":")
        if not separator or header_name.strip().lower() != name_bytes:
            continue
        values.append(header_value.strip().decode("ascii", errors="replace"))
    return values


def _has_raw_proxy_framing_header_rewrite(request: HTTPRequest) -> bool:
    for name in BODY_FRAMING_HEADERS:
        wire_values = _wire_header_values(request.wire_raw_bytes, name)
        current_values = request.headers.get_all(name, failobj=[])
        if wire_values != current_values:
            return True
    return False


def default_throttle_response(
    _: HTTPRequest,
    decision: ThrottleDecision,
) -> HTTPResponse:
    retry_after = max(1, math.ceil(decision.retry_after_seconds))
    return HTTPResponse.text(
        "Too Many Requests",
        status=429,
        headers={"Retry-After": str(retry_after)},
    )


@dataclass
class ThrottleMiddleware:
    throttler: RequestThrottler
    response: ThrottleResponseFunc = default_throttle_response

    def reset(self) -> None:
        self.throttler.reset()

    async def __call__(
        self,
        ctx: ResponderContext,
        call_next: ResponderNext,
    ) -> ResponseSpec:
        decision = self.throttler.check(ctx.request)
        if decision.allowed:
            return await call_next()
        return self.response(ctx.request, decision)


@dataclass
class ResponseSequenceMiddleware:
    responses: list[HTTPResponse]
    index: int = 0

    def reset(self) -> None:
        self.index = 0

    async def __call__(
        self,
        ctx: ResponderContext,
        call_next: ResponderNext,
    ) -> ResponseSpec:
        if self.index < len(self.responses):
            response = self.responses[self.index]
            self.index += 1
            return response
        return await call_next()


@dataclass
class HttpxForwardProxyMiddleware:
    client: httpx.AsyncClient

    async def __call__(
        self,
        ctx: ResponderContext,
        call_next: ResponderNext,
    ) -> ResponseSpec:
        if not ctx.request.is_proxy_request:
            return await call_next()
        return await forward_via_httpx(self.client, ctx.request)


@dataclass
class RawForwardProxyMiddleware:
    forwarder: Forwarder

    async def __call__(
        self,
        ctx: ResponderContext,
        call_next: ResponderNext,
    ) -> ResponseSpec:
        if not ctx.request.is_proxy_request:
            return await call_next()
        uri = ctx.request.target_uri
        if uri is None:
            return HTTPResponse.text(
                "Bad Request: Not an absolute URI",
                status=400,
            )

        wire_body_bytes = parsed_body_bytes_from_wire_raw_bytes(
            ctx.request.wire_raw_bytes
        )
        if (
            wire_body_bytes is not None
            and wire_body_bytes != ctx.request.body_bytes
        ):
            return HTTPResponse.text(
                "Raw proxy forwarding does not support request body rewrites.",
                status=500,
            )
        if _has_raw_proxy_framing_header_rewrite(ctx.request):
            return HTTPResponse.text(
                "Raw proxy forwarding does not support framing header "
                "rewrites.",
                status=500,
            )
        return ForwardProxyResponse(
            host=uri.host,
            port=uri.port,
            upstream_tls=(uri.scheme == "https"),
            request_wire_bytes=build_origin_form_request(ctx.request, uri),
            request_method=ctx.request.method,
            forwarder=self.forwarder,
        )


@dataclass
class RouterMiddleware:
    router: Router

    async def __call__(
        self,
        ctx: ResponderContext,
        call_next: ResponderNext,
    ) -> ResponseSpec:
        result = await self.router.handle(ctx)
        if result is None:
            return await call_next()
        return result


@dataclass
class HandlerMiddleware:
    get_handler: Callable[[], ResponderHandler | None]

    async def __call__(
        self,
        ctx: ResponderContext,
        call_next: ResponderNext,
    ) -> ResponseSpec:
        handler = self.get_handler()
        if handler is None:
            return await call_next()
        return await maybe_await(handler(ctx))
