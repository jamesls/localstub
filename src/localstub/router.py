from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from localstub.http.request import RecordedHTTPRequest
from localstub.http.utils import maybe_await
from localstub.middleware import (
    ResponderContext,
    ResponderNext,
    ResponseSpec,
    ensure_response_spec,
)

ResponderHandler = Callable[
    [ResponderContext],
    ResponseSpec | Awaitable[ResponseSpec],
]


@dataclass
class Router:
    _routes: dict[tuple[str, str], ResponderHandler] = field(
        default_factory=dict[tuple[str, str], ResponderHandler]
    )

    @property
    def has_routes(self) -> bool:
        return type(self) is not Router or bool(self._routes)

    def add(self, method: str, path: str, handler: ResponderHandler) -> None:
        self._routes[(method.upper(), path)] = handler

    def match(self, request: RecordedHTTPRequest) -> ResponderHandler | None:
        key = (
            request.method.upper() if request.method else "",
            request.effective_path,
        )
        return self._routes.get(key)

    async def handle(self, ctx: ResponderContext) -> ResponseSpec | None:
        handler = self.match(ctx.request)
        if handler is None:
            return None
        return ensure_response_spec(await maybe_await(handler(ctx)))


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
