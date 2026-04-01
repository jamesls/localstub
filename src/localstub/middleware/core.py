from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Protocol, TypeAlias, cast

from localstub.forward import Forwarder
from localstub.http.request import HTTPRequest, HTTPRequestHeaders
from localstub.http.response import RecordedResponse
from localstub.http.responsespec import HTTPResponse
from localstub.http.utils import maybe_await
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

    def clone_request(
        self,
        *,
        method: str | None = None,
        path: str | None = None,
        http_version: str | None = None,
        headers: Mapping[str, str] | None = None,
        body_bytes: bytes | None = None,
    ) -> ResponderContext:
        if (
            method is None
            and path is None
            and http_version is None
            and not headers
            and body_bytes is None
        ):
            return self

        request = self.request

        next_headers = (
            request.headers
            if not headers
            else request.headers.patch_set(headers)
        )

        if body_bytes is None:
            next_body_bytes = request.body_bytes
            next_body = request.body
        else:
            next_body_bytes = body_bytes
            next_body = body_bytes.decode("utf-8", errors="replace")

        next_request = replace(
            request,
            method=request.method if method is None else method,
            path=request.path if path is None else path,
            http_version=(
                request.http_version if http_version is None else http_version
            ),
            headers=next_headers,
            body=next_body,
            body_bytes=next_body_bytes,
        )
        return replace(self, request=next_request)


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
    *,
    capture_ctx: Callable[[ResponderContext], None] | None = None,
) -> Callable[[ResponderContext], Awaitable[ResponseSpec]]:
    async def app(ctx: ResponderContext) -> ResponseSpec:
        index = -1

        async def dispatch(
            i: int,
            current_ctx: ResponderContext,
        ) -> ResponseSpec:
            nonlocal index
            if i <= index:
                raise RuntimeError("call_next() called multiple times")
            index = i

            if i >= len(middlewares):
                result = await maybe_await(terminal(current_ctx))
                if capture_ctx is not None:
                    capture_ctx(current_ctx)
                return result

            mw = middlewares[i]
            delegated = False

            async def call_next(
                *,
                ctx: ResponderContext | None = None,
            ) -> ResponseSpec:
                nonlocal delegated
                delegated = True
                next_ctx = current_ctx if ctx is None else ctx
                return await dispatch(i + 1, next_ctx)

            result = await maybe_await(mw(current_ctx, call_next))
            if not delegated and capture_ctx is not None:
                capture_ctx(current_ctx)
            return result

        return await dispatch(0, ctx)

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
            current_ctx: SenderContext,
        ) -> SendResult:
            nonlocal index
            if i <= index:
                raise RuntimeError("call_next() called multiple times")
            index = i

            if i >= len(middlewares):
                return await maybe_await(terminal(current_ctx, resp))

            mw = middlewares[i]

            async def call_next(
                response: ResponseSpec,
                *,
                ctx: SenderContext | None = None,
            ) -> SendResult:
                next_ctx = current_ctx if ctx is None else ctx
                return await dispatch(i + 1, response, next_ctx)

            return await maybe_await(mw(current_ctx, resp, call_next))

        return await dispatch(0, response, ctx)

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
            current_ctx: HeaderContext,
        ) -> bool:
            nonlocal index
            if i <= index:
                raise RuntimeError("call_next() called multiple times")
            index = i

            if i >= len(middlewares):
                return cast(bool, await maybe_await(terminal(current_ctx)))

            mw = middlewares[i]

            async def call_next(
                *,
                ctx: HeaderContext | None = None,
            ) -> bool:
                next_ctx = current_ctx if ctx is None else ctx
                return await dispatch(i + 1, next_ctx)

            return cast(bool, await maybe_await(mw(current_ctx, call_next)))

        return await dispatch(0, ctx)

    return app
