from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Protocol, TypeAlias, cast

from localstub.forward import Forwarder
from localstub.http.request import HTTPRequest, HTTPRequestHeaders
from localstub.http.response import RecordedResponse
from localstub.http.responsespec import HTTPResponse
from localstub.throttle import Clock, MonotonicClock


class TimestampProvider(Protocol):
    def now(self) -> datetime: ...


class SystemTimestampProvider:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True)
class ConnectionMeta:
    client: tuple[str, int] | None


@dataclass(frozen=True)
class ServerServices:
    clock: Clock = field(default_factory=MonotonicClock)
    timestamp_provider: TimestampProvider = field(
        default_factory=SystemTimestampProvider
    )


@dataclass(frozen=True)
class ResponderContext:
    request: HTTPRequest
    connection: ConnectionMeta
    services: ServerServices
    state: dict[str, Any] = field(default_factory=dict)
    received_monotonic: float = 0.0

    def with_request(self, request: HTTPRequest) -> ResponderContext:
        return replace(self, request=request)


@dataclass(frozen=True)
class SenderContext:
    request: HTTPRequest
    connection: ConnectionMeta
    services: ServerServices
    state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForwardProxyResponse:
    host: str
    port: int
    upstream_tls: bool
    request_wire_bytes: bytes
    request_method: str | None
    forwarder: Forwarder


ResponseSpec: TypeAlias = HTTPResponse | ForwardProxyResponse


def _maybe_await(value: Any) -> Awaitable[Any]:
    if inspect.isawaitable(value):
        return cast(Awaitable[Any], value)

    async def done() -> Any:
        return value

    return done()


class ResponderNext(Protocol):
    def __call__(
        self,
        *,
        ctx: ResponderContext | None = None,
    ) -> Awaitable[ResponseSpec]: ...


ResponderMiddleware: TypeAlias = Callable[
    [ResponderContext, ResponderNext],
    ResponseSpec | Awaitable[ResponseSpec],
]


def compose_responder(
    middlewares: list[ResponderMiddleware],
    terminal: Callable[
        [ResponderContext], ResponseSpec | Awaitable[ResponseSpec]
    ],
) -> Callable[[ResponderContext], Awaitable[ResponseSpec]]:
    async def app(ctx: ResponderContext) -> ResponseSpec:
        index = -1

        async def dispatch(
            i: int,
            *,
            ctx_override: ResponderContext | None,
        ) -> ResponseSpec:
            nonlocal index
            if i <= index:
                raise RuntimeError("call_next() called multiple times")
            index = i

            next_ctx = ctx if ctx_override is None else ctx_override
            if i >= len(middlewares):
                return await _maybe_await(terminal(next_ctx))

            mw = middlewares[i]

            async def call_next(
                *,
                ctx: ResponderContext | None = None,
            ) -> ResponseSpec:
                return await dispatch(i + 1, ctx_override=ctx)

            return await _maybe_await(mw(next_ctx, call_next))

        return await dispatch(0, ctx_override=None)

    return app


@dataclass(frozen=True)
class SendResult:
    recorded: RecordedResponse
    should_close: bool


class SenderNext(Protocol):
    def __call__(
        self,
        response: ResponseSpec,
        *,
        ctx: SenderContext | None = None,
    ) -> Awaitable[SendResult]: ...


SenderMiddleware: TypeAlias = Callable[
    [SenderContext, ResponseSpec, SenderNext],
    SendResult | Awaitable[SendResult],
]


def compose_sender(
    middlewares: list[SenderMiddleware],
    terminal: Callable[
        [SenderContext, ResponseSpec], SendResult | Awaitable[SendResult]
    ],
) -> Callable[[SenderContext, ResponseSpec], Awaitable[SendResult]]:
    async def app(ctx: SenderContext, response: ResponseSpec) -> SendResult:
        index = -1

        async def dispatch(
            i: int,
            resp: ResponseSpec,
            *,
            ctx_override: SenderContext | None,
        ) -> SendResult:
            nonlocal index
            if i <= index:
                raise RuntimeError("call_next() called multiple times")
            index = i

            next_ctx = ctx if ctx_override is None else ctx_override
            if i >= len(middlewares):
                return await _maybe_await(terminal(next_ctx, resp))

            mw = middlewares[i]

            async def call_next(
                response: ResponseSpec,
                *,
                ctx: SenderContext | None = None,
            ) -> SendResult:
                return await dispatch(i + 1, response, ctx_override=ctx)

            return await _maybe_await(mw(next_ctx, resp, call_next))

        return await dispatch(0, response, ctx_override=None)

    return app


SendHeaderResponse = Callable[[HTTPResponse], Awaitable[None]]


@dataclass(frozen=True)
class HeaderContext:
    headers: HTTPRequestHeaders
    connection: ConnectionMeta
    services: ServerServices
    send: SendHeaderResponse
    state: dict[str, Any] = field(default_factory=dict)


class HeaderNext(Protocol):
    def __call__(
        self,
        *,
        ctx: HeaderContext | None = None,
    ) -> Awaitable[bool]: ...


HeaderMiddleware: TypeAlias = Callable[
    [HeaderContext, HeaderNext],
    bool | Awaitable[bool],
]


def compose_headers(
    middlewares: list[HeaderMiddleware],
    terminal: Callable[[HeaderContext], bool | Awaitable[bool]],
) -> Callable[[HeaderContext], Awaitable[bool]]:
    async def app(ctx: HeaderContext) -> bool:
        index = -1

        async def dispatch(
            i: int,
            *,
            ctx_override: HeaderContext | None,
        ) -> bool:
            nonlocal index
            if i <= index:
                raise RuntimeError("call_next() called multiple times")
            index = i

            next_ctx = ctx if ctx_override is None else ctx_override
            if i >= len(middlewares):
                return cast(bool, await _maybe_await(terminal(next_ctx)))

            mw = middlewares[i]

            async def call_next(
                *,
                ctx: HeaderContext | None = None,
            ) -> bool:
                return await dispatch(i + 1, ctx_override=ctx)

            return cast(bool, await _maybe_await(mw(next_ctx, call_next)))

        return await dispatch(0, ctx_override=None)

    return app
