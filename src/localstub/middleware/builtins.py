from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from localstub.forward import RawForwarder
from localstub.http.client import HTTPClient
from localstub.http.proxy import (
    build_origin_form_request,
    forward_proxy_request,
)
from localstub.http.request import RecordedHTTPRequest
from localstub.http.responsespec import HTTPResponse
from localstub.http.utils import maybe_await
from localstub.middleware.core import (
    ForwardProxyResponse,
    ResponderContext,
    ResponderMiddleware,
    ResponderNext,
    ResponseSpec,
)
from localstub.router import ResponderHandler, Router
from localstub.throttle import RequestThrottler, ThrottleDecision

ThrottleResponseFunc = Callable[
    [RecordedHTTPRequest, ThrottleDecision], HTTPResponse
]


def _has_framing_header_rewrite(recorded: RecordedHTTPRequest) -> bool:
    return any(
        recorded.request.headers.get_all(name)
        != recorded.as_received.headers.get_all(name)
        for name in ("Content-Length", "Transfer-Encoding")
    )


def default_throttle_response(
    _: RecordedHTTPRequest,
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
class ForwardProxyMiddleware:
    client: HTTPClient

    async def __call__(
        self,
        ctx: ResponderContext,
        call_next: ResponderNext,
    ) -> ResponseSpec:
        if not ctx.request.is_proxy_request:
            return await call_next()
        return await forward_proxy_request(self.client, ctx.request)


@dataclass
class RawForwardProxyMiddleware:
    forwarder: RawForwarder

    async def __call__(
        self,
        ctx: ResponderContext,
        call_next: ResponderNext,
    ) -> ResponseSpec:
        if not ctx.request.is_proxy_request:
            return await call_next()
        recorded = ctx.request
        uri = recorded.target_uri
        if uri is None:
            return HTTPResponse.text(
                "Bad Request: Not an absolute URI",
                status=400,
            )

        if recorded.request.body != recorded.as_received.body:
            return HTTPResponse.text(
                "Raw proxy forwarding does not support request body rewrites.",
                status=500,
            )
        if _has_framing_header_rewrite(recorded):
            return HTTPResponse.text(
                "Raw proxy forwarding does not support framing header "
                "rewrites.",
                status=500,
            )
        try:
            request_wire_bytes = build_origin_form_request(recorded, uri)
        except ValueError as exc:
            return HTTPResponse.text(f"Bad Request: {exc}", status=400)

        return ForwardProxyResponse(
            host=uri.host,
            port=uri.port,
            upstream_tls=(uri.scheme == "https"),
            request_wire_bytes=request_wire_bytes,
            request_method=recorded.method,
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


type SlotName = Literal["throttle", "sequence", "raw_proxy", "proxy"]

_SLOT_ORDER: tuple[SlotName, ...] = (
    "throttle",
    "sequence",
    "raw_proxy",
    "proxy",
)


@runtime_checkable
class SupportsReset(Protocol):
    def reset(self) -> None: ...


class BuiltinMiddlewares:
    """Ordered slots for a server's built-in responder middlewares.

    Slot order is pipeline priority order.
    """

    def __init__(self) -> None:
        self._slots: dict[SlotName, ResponderMiddleware | None] = {
            name: None for name in _SLOT_ORDER
        }

    def set(self, name: SlotName, middleware: ResponderMiddleware) -> None:
        self._slots[name] = middleware

    def clear(self, name: SlotName) -> None:
        self._slots[name] = None

    def active(self) -> list[ResponderMiddleware]:
        return [m for m in self._slots.values() if m is not None]

    @property
    def any_active(self) -> bool:
        return any(m is not None for m in self._slots.values())

    def reset_all(self) -> None:
        for middleware in self.active():
            if isinstance(middleware, SupportsReset):
                middleware.reset()
