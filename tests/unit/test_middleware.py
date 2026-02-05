from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import pytest

from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest, HTTPRequestHeaders
from localstub.http.response import RecordedResponse
from localstub.http.responsespec import HTTPResponse
from localstub.middleware import (
    ConnectionMeta,
    HeaderContext,
    ResponderContext,
    SenderContext,
    SendResult,
    ServerServices,
    TimestampProvider,
    compose_headers,
    compose_responder,
    compose_sender,
)


@dataclass(frozen=True)
class FixedTimestampProvider(TimestampProvider):
    value: datetime

    def now(self) -> datetime:
        return self.value


def _services() -> ServerServices:
    return ServerServices(
        timestamp_provider=FixedTimestampProvider(
            datetime(2026, 1, 1, tzinfo=timezone.utc)
        )
    )


@pytest.mark.asyncio
async def test_compose_responder_onion_order() -> None:
    order: list[str] = []

    async def mw_a(ctx: ResponderContext, call_next) -> HTTPResponse:
        order.append("a:in")
        result = await call_next()
        order.append("a:out")
        assert isinstance(result, HTTPResponse)
        return result

    async def mw_b(ctx: ResponderContext, call_next) -> HTTPResponse:
        order.append("b:in")
        result = await call_next()
        order.append("b:out")
        assert isinstance(result, HTTPResponse)
        return result

    async def terminal(_: ResponderContext) -> HTTPResponse:
        order.append("terminal")
        return HTTPResponse.text("ok")

    app = compose_responder([mw_a, mw_b], terminal)
    ctx = ResponderContext(
        request=HTTPRequest(method="GET", path="/", http_version="1.1"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    result = await app(ctx)
    assert isinstance(result, HTTPResponse)
    assert result.body == b"ok"
    assert order == ["a:in", "b:in", "terminal", "b:out", "a:out"]


@pytest.mark.asyncio
async def test_compose_responder_ctx_override_rewrites_request() -> None:
    async def rewrite_path(ctx: ResponderContext, call_next) -> HTTPResponse:
        rewritten = ctx.request.with_path("/rewritten")
        return await call_next(ctx=ctx.with_request(rewritten))

    async def terminal(ctx: ResponderContext) -> HTTPResponse:
        return HTTPResponse.text(ctx.request.path)

    app = compose_responder([rewrite_path], terminal)
    ctx = ResponderContext(
        request=HTTPRequest(method="GET", path="/orig", http_version="1.1"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    result = await app(ctx)
    assert isinstance(result, HTTPResponse)
    assert result.body == b"/rewritten"


@pytest.mark.asyncio
async def test_responder_context_clone_request_patches_request() -> None:
    async def mw(ctx: ResponderContext, call_next) -> HTTPResponse:
        ctx2 = ctx.clone_request(
            headers={"X-Request-Id": "test-123"},
            body_bytes=b"corrupt",
        )
        assert "X-Request-Id" not in ctx.request.headers
        return await call_next(ctx=ctx2)

    async def terminal(ctx: ResponderContext) -> HTTPResponse:
        rid = ctx.request.headers["X-Request-Id"]
        return HTTPResponse.text(f"{rid}:{ctx.request.body}")

    app = compose_responder([mw], terminal)
    ctx = ResponderContext(
        request=HTTPRequest(method="POST", path="/", http_version="1.1"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    result = await app(ctx)
    assert isinstance(result, HTTPResponse)
    assert result.body == b"test-123:corrupt"


@pytest.mark.asyncio
async def test_compose_responder_call_next_twice_raises() -> None:
    async def mw(ctx: ResponderContext, call_next) -> HTTPResponse:
        await call_next()
        await call_next()
        raise AssertionError("unreachable")

    async def terminal(_: ResponderContext) -> HTTPResponse:
        return HTTPResponse.text("ok")

    app = compose_responder([mw], terminal)
    ctx = ResponderContext(
        request=HTTPRequest(method="GET", path="/", http_version="1.1"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    with pytest.raises(
        RuntimeError, match="call_next\\(\\) called multiple times"
    ):
        await app(ctx)


@pytest.mark.asyncio
async def test_compose_sender_call_next_twice_raises() -> None:
    async def mw(ctx: SenderContext, response, call_next) -> SendResult:
        _ = ctx
        await call_next(response)
        await call_next(response)
        raise AssertionError("unreachable")

    async def terminal(ctx: SenderContext, response) -> SendResult:
        _ = (ctx, response)
        return SendResult(
            recorded=RecordedResponse(
                status=200,
                reason="OK",
                headers=None,
                body=None,
                wire_raw_bytes=b"",
            ),
            should_close=False,
        )

    app = compose_sender([mw], terminal)

    ctx = SenderContext(
        request=HTTPRequest(method="GET", path="/", http_version="1.1"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    with pytest.raises(
        RuntimeError, match="call_next\\(\\) called multiple times"
    ):
        await app(ctx, HTTPResponse.text("ok"))


@pytest.mark.asyncio
async def test_compose_headers_short_circuit() -> None:
    seen: list[str] = []

    async def mw_a(ctx: HeaderContext, call_next) -> bool:
        seen.append("a")
        return await call_next()

    async def mw_stop(ctx: HeaderContext, call_next) -> bool:
        _ = call_next
        seen.append("stop")
        await ctx.send(HTTPResponse(status=100))
        return False

    async def terminal(_: HeaderContext) -> bool:
        seen.append("terminal")
        return True

    app = compose_headers([mw_a, mw_stop], terminal)

    async def send(_: HTTPResponse) -> None:
        seen.append("send")

    ctx = HeaderContext(
        headers=HTTPRequestHeaders(
            method="GET",
            path="/",
            http_version="1.1",
            headers=Headers.empty(),
            wire_raw_bytes=b"GET / HTTP/1.1\r\n\r\n",
        ),
        connection=ConnectionMeta(client=None),
        services=_services(),
        send=send,
    )

    should_continue = await app(ctx)
    assert should_continue is False
    assert seen == ["a", "stop", "send"]
