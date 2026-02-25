from __future__ import annotations

import pytest

from localstub.http.request import HTTPRequest
from localstub.http.responsespec import HTTPResponse
from localstub.middleware import (
    ConnectionMeta,
    ResponderContext,
    ServerServices,
)
from localstub.router import Router


def _ctx(method: str, path: str) -> ResponderContext:
    return ResponderContext(
        request=HTTPRequest(method=method, path=path, http_version="1.1"),
        connection=ConnectionMeta(client=None),
        services=ServerServices(),
    )


@pytest.mark.asyncio
async def test_router_handle_no_match_returns_none() -> None:
    router = Router()
    assert await router.handle(_ctx("GET", "/")) is None


@pytest.mark.asyncio
async def test_router_handle_matched_handler_returning_none_raises() -> None:
    router = Router()

    def handler(_: ResponderContext):
        return None

    router.add("GET", "/", handler)

    with pytest.raises(TypeError, match="Unhandled response spec: NoneType"):
        await router.handle(_ctx("GET", "/"))


@pytest.mark.asyncio
async def test_router_handle_async_handler_returning_none_raises() -> None:
    router = Router()

    async def handler(_: ResponderContext):
        return None

    router.add("GET", "/", handler)

    with pytest.raises(TypeError, match="Unhandled response spec: NoneType"):
        await router.handle(_ctx("GET", "/"))


@pytest.mark.asyncio
async def test_router_handle_returns_response_spec() -> None:
    router = Router()

    def handler(_: ResponderContext) -> HTTPResponse:
        return HTTPResponse.text("ok")

    router.add("GET", "/", handler)

    result = await router.handle(_ctx("GET", "/"))
    assert isinstance(result, HTTPResponse)
    assert result.body == b"ok"
