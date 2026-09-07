from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum, auto
from typing import Any, Protocol

from localstub.clock import Clock, MonotonicClock
from localstub.forward import RawForwarder
from localstub.http.request import HTTPRequestHeaders, RecordedHTTPRequest
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HTTPResponse
from localstub.http.utils import maybe_await


class _Unset(Enum):
    """Sentinel distinguishing an omitted argument from an explicit None."""

    TOKEN = auto()


_UNSET = _Unset.TOKEN


class TimestampProvider(Protocol):
    def now(self) -> datetime: ...


class SystemTimestampProvider:
    def now(self) -> datetime:
        return datetime.now(UTC)


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
    request: RecordedHTTPRequest
    connection: ConnectionMeta
    services: ServerServices
    state: dict[str, Any] = field(default_factory=dict[str, Any])
    received_monotonic: float = 0.0

    def with_request(
        self,
        request: RecordedHTTPRequest,
    ) -> ResponderContext:
        return replace(self, request=request)

    def clone_request(
        self,
        *,
        method: str | None = None,
        target: str | None = None,
        headers: Mapping[str, str] | None = None,
        body: bytes | _Unset | None = _UNSET,
    ) -> ResponderContext:
        """Return a copy of this context with request fields replaced.

        ``body=None`` explicitly clears the body (no content); omitting
        ``body`` keeps the existing one.
        """
        if (
            method is None
            and target is None
            and not headers
            and body is _UNSET
        ):
            return self

        recorded = self.request
        request = recorded.request

        next_headers = (
            request.headers
            if not headers
            else request.headers.patch_set(headers)
        )

        next_request = replace(
            request,
            method=request.method if method is None else method,
            target=request.target if target is None else target,
            headers=next_headers,
            body=request.body if body is _UNSET else body,
        )
        return replace(self, request=replace(recorded, request=next_request))


@dataclass(frozen=True)
class SenderContext:
    request: RecordedHTTPRequest
    connection: ConnectionMeta
    services: ServerServices
    state: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass(frozen=True)
class ForwardProxyResponse:
    host: str
    port: int
    upstream_tls: bool
    request_wire_bytes: bytes
    request_method: str | None
    forwarder: RawForwarder


type ResponseSpec = HTTPResponse | ForwardProxyResponse


def ensure_response_spec(value: object) -> ResponseSpec:
    """Validate a handler's return value at the type-system boundary.

    Handlers are annotated to return a ``ResponseSpec`` but user code may
    return anything at runtime, so the check is performed against
    ``object`` rather than trusting the annotation.
    """
    if isinstance(value, HTTPResponse | ForwardProxyResponse):
        return value
    raise TypeError(f"Unhandled response spec: {type(value).__name__}")


class ResponderNext(Protocol):
    def __call__(
        self,
        *,
        ctx: ResponderContext | None = None,
    ) -> Awaitable[ResponseSpec]: ...


type ResponderMiddleware = Callable[
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
    recorded: RecordedHTTPResponse
    should_close: bool


class SenderNext(Protocol):
    def __call__(
        self,
        response: ResponseSpec,
        *,
        ctx: SenderContext | None = None,
    ) -> Awaitable[SendResult]: ...


type SenderMiddleware = Callable[
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
    state: dict[str, Any] = field(default_factory=dict[str, Any])


class HeaderNext(Protocol):
    def __call__(
        self,
        *,
        ctx: HeaderContext | None = None,
    ) -> Awaitable[bool]: ...


type HeaderMiddleware = Callable[
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
                return await maybe_await(terminal(current_ctx))

            mw = middlewares[i]

            async def call_next(
                *,
                ctx: HeaderContext | None = None,
            ) -> bool:
                next_ctx = current_ctx if ctx is None else ctx
                return await dispatch(i + 1, next_ctx)

            return await maybe_await(mw(current_ctx, call_next))

        return await dispatch(0, ctx)

    return app
