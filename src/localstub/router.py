from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import cast

from localstub.http.request import HTTPRequest
from localstub.middleware import ResponderContext, ResponseSpec

ResponderHandler = Callable[
    [ResponderContext],
    ResponseSpec | Awaitable[ResponseSpec],
]


@dataclass
class Router:
    _routes: dict[tuple[str, str], ResponderHandler] = field(
        default_factory=dict
    )

    def add(self, method: str, path: str, handler: ResponderHandler) -> None:
        self._routes[(method.upper(), path)] = handler

    def match(self, request: HTTPRequest) -> ResponderHandler | None:
        key = (
            request.method.upper() if request.method else "",
            request.effective_path,
        )
        return self._routes.get(key)

    async def handle(self, ctx: ResponderContext) -> ResponseSpec | None:
        handler = self.match(ctx.request)
        if handler is None:
            return None
        result = handler(ctx)
        if inspect.isawaitable(result):
            return await cast(Awaitable[ResponseSpec], result)
        return cast(ResponseSpec, result)
