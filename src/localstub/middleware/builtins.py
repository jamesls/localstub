from __future__ import annotations

import inspect
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

import httpx

from localstub.forward import Forwarder
from localstub.http.connection import should_close_connection
from localstub.http.proxy import build_origin_form_request, forward_via_httpx
from localstub.http.request import HTTPRequest
from localstub.http.response import RecordedResponse
from localstub.http.responsespec import HTTPResponse
from localstub.middleware.core import (
    ForwardProxyResponse,
    ResponderContext,
    ResponderNext,
    ResponseSpec,
    SenderContext,
    SenderNext,
    SendResult,
)
from localstub.router import ResponderHandler, Router
from localstub.throttle import RequestThrottler, ThrottleDecision

ThrottleResponseFunc = Callable[[HTTPRequest, ThrottleDecision], HTTPResponse]


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
        result = handler(ctx)
        if inspect.isawaitable(result):
            return await cast(Awaitable[ResponseSpec], result)
        return cast(ResponseSpec, result)


@dataclass
class RawForwardProxySender:
    async def __call__(
        self,
        ctx: SenderContext,
        response: ResponseSpec,
        call_next: SenderNext,
    ) -> SendResult:
        if not isinstance(response, ForwardProxyResponse):
            return await call_next(response)

        writer = ctx.conn.writer
        wire_offset = len(writer.bytes_sent)
        result = await response.forwarder.forward_and_relay(
            host=response.host,
            port=response.port,
            request_wire_bytes=response.request_wire_bytes,
            client_writer=cast(Any, writer),
            request_method=response.request_method,
            upstream_tls=response.upstream_tls,
        )
        wire_bytes = writer.bytes_sent[wire_offset:]

        if result is None:
            return await call_next(
                HTTPResponse.text("Bad Gateway", status=502),
            )

        recorded = RecordedResponse(
            status=result.status,
            reason=result.reason,
            headers=result.headers,
            body=result.body.decode("utf-8", errors="replace"),
            wire_raw_bytes=wire_bytes,
        )
        return SendResult(
            recorded=recorded,
            should_close=should_close_connection(
                ctx.request,
                response_headers=result.headers,
            ),
        )
