from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

import pytest

from localstub.http.headers import Headers
from localstub.http.request import (
    HTTPRequest,
    HTTPRequestHeaders,
    RecordedHTTPRequest,
)
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HTTPResponse
from localstub.middleware import (
    ConnectionMeta,
    HeaderContext,
    ResponderContext,
    ResponseSpec,
    SenderContext,
    SendResult,
    ServerServices,
    TimestampProvider,
    compose_headers,
    compose_responder,
    compose_sender,
)
from localstub.middleware.builtins import (
    BuiltinMiddlewares,
    ForwardProxyMiddleware,
    ResponseSequenceMiddleware,
)


@dataclass(frozen=True)
class FixedTimestampProvider(TimestampProvider):
    value: datetime

    def now(self) -> datetime:
        return self.value


class StubHTTPClient:
    def __init__(self, response: HTTPResponse) -> None:
        self.response = response
        self.requests: list[HTTPRequest] = []

    async def send(self, request: HTTPRequest) -> HTTPResponse:
        self.requests.append(request)
        return self.response


def _services() -> ServerServices:
    return ServerServices(
        timestamp_provider=FixedTimestampProvider(
            datetime(2026, 1, 1, tzinfo=UTC)
        )
    )


def _recorded(
    method: str,
    target: str,
    *,
    headers: Headers | None = None,
    body: bytes | None = None,
) -> RecordedHTTPRequest:
    request = HTTPRequest(
        method=method,
        target=target,
        headers=headers if headers is not None else Headers.empty(),
        body=body,
    )
    return RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=f"{method} {target} HTTP/1.1\r\n\r\n".encode(),
        http_version="1.1",
    )


def _recorded_response() -> RecordedHTTPResponse:
    return RecordedHTTPResponse(
        response=HTTPResponse(status=200),
        reason="OK",
        wire_raw_bytes=b"",
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
        request=_recorded("GET", "/"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    result = await app(ctx)
    assert isinstance(result, HTTPResponse)
    assert result.body == b"ok"
    assert order == ["a:in", "b:in", "terminal", "b:out", "a:out"]


@pytest.mark.asyncio
async def test_compose_responder_ctx_override_rewrites_request() -> None:
    async def rewrite_target(ctx: ResponderContext, call_next) -> HTTPResponse:
        rewritten = ctx.request.with_target("/rewritten")
        return await call_next(ctx=ctx.with_request(rewritten))

    async def terminal(ctx: ResponderContext) -> HTTPResponse:
        return HTTPResponse.text(ctx.request.target)

    app = compose_responder([rewrite_target], terminal)
    ctx = ResponderContext(
        request=_recorded("GET", "/orig"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    result = await app(ctx)
    assert isinstance(result, HTTPResponse)
    assert result.body == b"/rewritten"


@pytest.mark.asyncio
async def test_responder_ctx_override_persists_across_chain() -> None:
    seen_targets: list[str] = []

    async def rewrite(ctx: ResponderContext, call_next) -> HTTPResponse:
        rewritten = ctx.request.with_target("/rewritten")
        return await call_next(ctx=ctx.with_request(rewritten))

    async def observe(ctx: ResponderContext, call_next) -> HTTPResponse:
        seen_targets.append(ctx.request.target)
        return await call_next()

    async def terminal(ctx: ResponderContext) -> HTTPResponse:
        seen_targets.append(ctx.request.target)
        return HTTPResponse.text(ctx.request.target)

    app = compose_responder([rewrite, observe], terminal)
    ctx = ResponderContext(
        request=_recorded("GET", "/orig"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    result = await app(ctx)
    assert isinstance(result, HTTPResponse)
    assert result.body == b"/rewritten"
    assert seen_targets == ["/rewritten", "/rewritten"]


@pytest.mark.asyncio
async def test_responder_context_clone_request_patches_request() -> None:
    async def mw(ctx: ResponderContext, call_next) -> HTTPResponse:
        ctx2 = ctx.clone_request(
            headers={"X-Request-Id": "test-123"},
            body=b"corrupt",
        )
        assert "X-Request-Id" not in ctx.request.headers
        return await call_next(ctx=ctx2)

    async def terminal(ctx: ResponderContext) -> HTTPResponse:
        rid = ctx.request.headers["X-Request-Id"]
        return HTTPResponse.text(f"{rid}:{ctx.request.text}")

    app = compose_responder([mw], terminal)
    ctx = ResponderContext(
        request=_recorded("POST", "/"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    result = await app(ctx)
    assert isinstance(result, HTTPResponse)
    assert result.body == b"test-123:corrupt"


def test_responder_context_clone_request_keeps_wire_evidence() -> None:
    recorded = _recorded("POST", "/orig", body=b"payload")
    ctx = ResponderContext(
        request=recorded,
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    ctx2 = ctx.clone_request(target="/rewritten", body=b"other")

    assert ctx2.request.target == "/rewritten"
    assert ctx2.request.body == b"other"
    assert ctx2.request.as_received == recorded.as_received
    assert ctx2.request.wire_raw_bytes == recorded.wire_raw_bytes
    assert ctx2.request.http_version == "1.1"


def test_responder_context_clone_request_body_none_clears_body() -> None:
    recorded = _recorded("POST", "/orig", body=b"payload")
    ctx = ResponderContext(
        request=recorded,
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    ctx2 = ctx.clone_request(body=None)

    assert ctx2.request.body is None
    assert ctx2.request.as_received.body == b"payload"


def test_responder_context_clone_request_no_changes_returns_self() -> None:
    ctx = ResponderContext(
        request=_recorded("GET", "/"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    assert ctx.clone_request() is ctx


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
        request=_recorded("GET", "/"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    with pytest.raises(
        RuntimeError, match="call_next\\(\\) called multiple times"
    ):
        await app(ctx)


@pytest.mark.asyncio
async def test_forward_proxy_middleware_forwards_proxy_request() -> None:
    client = StubHTTPClient(HTTPResponse.text("upstream"))
    middleware = ForwardProxyMiddleware(client=client)
    ctx = ResponderContext(
        request=_recorded(
            "GET",
            "http://example.com/upstream",
            headers=Headers.from_items([("Host", "example.com")]),
        ),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    async def call_next(
        *,
        ctx: ResponderContext | None = None,
    ) -> ResponseSpec:
        raise AssertionError("unreachable")

    result = await middleware(ctx, call_next)
    assert isinstance(result, HTTPResponse)
    assert result.status == 200
    assert result.body == b"upstream"
    assert len(client.requests) == 1
    sent = client.requests[0]
    assert sent.method == "GET"
    assert sent.target == "http://example.com/upstream"
    assert "Host" not in sent.headers


@pytest.mark.asyncio
async def test_forward_proxy_middleware_delegates_non_proxy_request() -> None:
    client = StubHTTPClient(HTTPResponse.text("upstream"))
    middleware = ForwardProxyMiddleware(client=client)
    ctx = ResponderContext(
        request=_recorded("GET", "/local"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )
    local = HTTPResponse.text("local")

    async def call_next(
        *,
        ctx: ResponderContext | None = None,
    ) -> ResponseSpec:
        return local

    result = await middleware(ctx, call_next)
    assert result is local
    assert client.requests == []


def test_builtin_middlewares_active_returns_slot_priority_order() -> None:
    builtins = BuiltinMiddlewares()
    proxy = ForwardProxyMiddleware(StubHTTPClient(HTTPResponse.json({})))
    sequence = ResponseSequenceMiddleware([HTTPResponse.json({})])
    builtins.set("proxy", proxy)
    builtins.set("sequence", sequence)

    assert builtins.active() == [sequence, proxy]


def test_builtin_middlewares_clear_empties_slot() -> None:
    builtins = BuiltinMiddlewares()
    builtins.set("sequence", ResponseSequenceMiddleware([]))
    assert builtins.any_active

    builtins.clear("sequence")

    assert not builtins.any_active
    assert builtins.active() == []


def test_builtin_middlewares_reset_all_skips_non_resettable() -> None:
    builtins = BuiltinMiddlewares()
    sequence = ResponseSequenceMiddleware([HTTPResponse.json({})], index=1)
    builtins.set("sequence", sequence)
    builtins.set(
        "proxy",
        ForwardProxyMiddleware(StubHTTPClient(HTTPResponse.json({}))),
    )

    builtins.reset_all()

    assert sequence.index == 0


@pytest.mark.asyncio
async def test_sender_ctx_override_persists_across_chain() -> None:
    seen_targets: list[str] = []

    async def rewrite(
        ctx: SenderContext,
        response: ResponseSpec,
        call_next,
    ) -> SendResult:
        next_ctx = replace(ctx, request=ctx.request.with_target("/rewritten"))
        return await call_next(response, ctx=next_ctx)

    async def observe(
        ctx: SenderContext,
        response: ResponseSpec,
        call_next,
    ) -> SendResult:
        seen_targets.append(ctx.request.target)
        return await call_next(response)

    async def terminal(
        ctx: SenderContext,
        response: ResponseSpec,
    ) -> SendResult:
        _ = response
        seen_targets.append(ctx.request.target)
        return SendResult(
            recorded=_recorded_response(),
            should_close=False,
        )

    app = compose_sender([rewrite, observe], terminal)
    ctx = SenderContext(
        request=_recorded("GET", "/orig"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    result = await app(ctx, HTTPResponse.text("ok"))
    assert not result.should_close
    assert seen_targets == ["/rewritten", "/rewritten"]


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
            recorded=_recorded_response(),
            should_close=False,
        )

    app = compose_sender([mw], terminal)

    ctx = SenderContext(
        request=_recorded("GET", "/"),
        connection=ConnectionMeta(client=None),
        services=_services(),
    )

    with pytest.raises(
        RuntimeError, match="call_next\\(\\) called multiple times"
    ):
        await app(ctx, HTTPResponse.text("ok"))


@pytest.mark.asyncio
async def test_headers_ctx_override_persists_across_chain() -> None:
    seen_paths: list[str | None] = []

    async def rewrite(ctx: HeaderContext, call_next) -> bool:
        next_headers = replace(ctx.headers, path="/rewritten")
        return await call_next(ctx=replace(ctx, headers=next_headers))

    async def observe(ctx: HeaderContext, call_next) -> bool:
        seen_paths.append(ctx.headers.path)
        return await call_next()

    async def terminal(ctx: HeaderContext) -> bool:
        seen_paths.append(ctx.headers.path)
        return True

    app = compose_headers([rewrite, observe], terminal)

    async def send(_: HTTPResponse) -> None:
        pass

    ctx = HeaderContext(
        headers=HTTPRequestHeaders(
            method="GET",
            path="/orig",
            http_version="1.1",
            headers=Headers.empty(),
            wire_raw_bytes=b"GET /orig HTTP/1.1\r\n\r\n",
        ),
        connection=ConnectionMeta(client=None),
        services=_services(),
        send=send,
    )

    should_continue = await app(ctx)
    assert should_continue
    assert seen_paths == ["/rewritten", "/rewritten"]


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
    assert not should_continue
    assert seen == ["a", "stop", "send"]
