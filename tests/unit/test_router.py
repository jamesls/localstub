from __future__ import annotations

import pytest

from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.http.responsespec import HTTPResponse
from localstub.middleware import (
    ConnectionMeta,
    ResponderContext,
    ServerServices,
)
from localstub.router import Router


def _recorded(method: str, target: str) -> RecordedHTTPRequest:
    request = HTTPRequest(method=method, target=target)
    return RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=f"{method} {target} HTTP/1.1\r\n\r\n".encode(),
        http_version="1.1",
    )


def _ctx(method: str, target: str) -> ResponderContext:
    return ResponderContext(
        request=_recorded(method, target),
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


def test_router_match_uses_effective_path_for_absolute_target() -> None:
    router = Router()

    def handler(_: ResponderContext) -> HTTPResponse:
        return HTTPResponse.text("ok")

    router.add("GET", "/foo", handler)

    matched = router.match(_recorded("GET", "http://example.com/foo"))
    assert matched is handler
