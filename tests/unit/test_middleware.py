from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from unittest.mock import create_autospec

import pytest

from localstub.forward import RawForwarder
from localstub.http.headers import Headers
from localstub.http.request import (
    HTTPRequest,
    HTTPRequestHeaders,
    RecordedHTTPRequest,
)
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HTTPResponse
from localstub.middleware import (
    CloseConnection,
    CloseDuringRequest,
    ConnectionMeta,
    ForwardProxyResponse,
    HeaderContext,
    HeaderDecision,
    HeaderNext,
    ResponderContext,
    ResponderNext,
    ResponseSpec,
    SenderContext,
    SendResult,
    ServerServices,
    TimestampProvider,
    compose_headers,
    compose_responder,
    compose_sender,
    ensure_response_spec,
)
from localstub.middleware.builtins import (
    BuiltinMiddlewares,
    ForwardProxyMiddleware,
    RawForwardProxyMiddleware,
    ResponseSequenceMiddleware,
    ThrottleMiddleware,
    close_during_request,
)
from localstub.throttle import ThrottleDecision, TokenBucketThrottler


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

    async def aclose(self) -> None:
        pass


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


class _FixedClock:
    def now(self) -> float:
        return 0.0


def _responder_ctx(recorded: RecordedHTTPRequest) -> ResponderContext:
    return ResponderContext(
        request=recorded,
        connection=ConnectionMeta(client=None),
        services=_services(),
    )


def _header_ctx() -> HeaderContext:
    async def send(_: HTTPResponse) -> None:
        pass

    return HeaderContext(
        headers=HTTPRequestHeaders(
            method="POST",
            path="/upload",
            http_version="1.1",
            headers=Headers.empty(),
            wire_raw_bytes=b"POST /upload HTTP/1.1\r\n\r\n",
        ),
        connection=ConnectionMeta(client=None),
        services=_services(),
        send=send,
    )


async def _never_next(
    *,
    ctx: ResponderContext | None = None,
) -> ResponseSpec:
    raise AssertionError("call_next() must not be called")


async def _ok_next(
    *,
    ctx: ResponderContext | None = None,
) -> ResponseSpec:
    return HTTPResponse.text("ok")


async def _allow(_: HeaderContext) -> HeaderDecision:
    return True


async def _refuse(_: HeaderContext) -> HeaderDecision:
    raise AssertionError("header terminal must not run")


async def _refuse_responder(_: ResponderContext) -> ResponseSpec:
    raise AssertionError("responder terminal must not run")


async def _ok_terminal(_: ResponderContext) -> ResponseSpec:
    return HTTPResponse.text("ok")


def _throttler(rate_per_second: float = 1.0) -> TokenBucketThrottler:
    return TokenBucketThrottler(
        rate_per_second=rate_per_second,
        key=lambda _: "global",
        clock=_FixedClock(),
    )


def _raw_forwarder() -> RawForwarder:
    return create_autospec(RawForwarder, instance=True)


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
async def test_compose_responder_capture_ctx_on_terminal() -> None:
    captured: list[ResponderContext] = []

    async def rewrite(
        ctx: ResponderContext, call_next: ResponderNext
    ) -> ResponseSpec:
        return await call_next(ctx=ctx.clone_request(target="/rewritten"))

    app = compose_responder(
        [rewrite], _ok_terminal, capture_ctx=captured.append
    )

    await app(_responder_ctx(_recorded("GET", "/orig")))

    assert [ctx.request.target for ctx in captured] == ["/rewritten"]


@pytest.mark.asyncio
async def test_compose_responder_capture_ctx_on_short_circuit() -> None:
    captured: list[ResponderContext] = []

    async def short_circuit(
        ctx: ResponderContext, call_next: ResponderNext
    ) -> ResponseSpec:
        _ = (ctx, call_next)
        return CloseConnection()

    app = compose_responder(
        [short_circuit], _refuse_responder, capture_ctx=captured.append
    )
    ctx = _responder_ctx(_recorded("GET", "/orig"))

    result = await app(ctx)

    assert isinstance(result, CloseConnection)
    assert len(captured) == 1
    assert captured[0] is ctx


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


def test_close_connection_defaults_to_graceful_close_without_delay() -> None:
    close = CloseConnection()

    assert not close.reset
    assert close.delay == pytest.approx(0.0)


def test_close_connection_negative_delay_raises_value_error() -> None:
    with pytest.raises(ValueError, match="delay must be non-negative"):
        CloseConnection(delay=-0.1)


def test_close_connection_compares_by_value_and_is_hashable() -> None:
    close = CloseConnection(reset=True, delay=0.5)

    assert close == CloseConnection(reset=True, delay=0.5)
    assert close != CloseConnection(reset=True)
    assert len({close, CloseConnection(reset=True, delay=0.5)}) == 1


def test_close_during_request_defaults_close_after_headers() -> None:
    decision = CloseDuringRequest()

    assert decision.after_body_bytes == 0
    assert not decision.reset


def test_close_during_request_negative_threshold_raises_value_error() -> None:
    with pytest.raises(
        ValueError, match="after_body_bytes must be non-negative"
    ):
        CloseDuringRequest(after_body_bytes=-1)


def test_close_during_request_bool_raises_type_error() -> None:
    with pytest.raises(TypeError, match="Return HeaderDecision unchanged"):
        bool(CloseDuringRequest())


def test_close_during_request_not_operator_raises_type_error() -> None:
    decision: HeaderDecision = CloseDuringRequest(after_body_bytes=10)

    with pytest.raises(TypeError, match="Return HeaderDecision unchanged"):
        _ = not decision


def test_close_during_request_compares_by_value_and_is_hashable() -> None:
    decision = CloseDuringRequest(after_body_bytes=64, reset=True)

    assert decision == CloseDuringRequest(after_body_bytes=64, reset=True)
    assert decision != CloseDuringRequest(after_body_bytes=64)
    assert len({decision, CloseDuringRequest(64, True)}) == 1


def test_ensure_response_spec_returns_close_connection_unchanged() -> None:
    close = CloseConnection(reset=True)

    assert ensure_response_spec(close) is close


def test_ensure_response_spec_rejects_unknown_value() -> None:
    with pytest.raises(TypeError, match="Unhandled response spec: str"):
        ensure_response_spec("not a response")


def test_send_result_defaults_closed_to_none() -> None:
    result = SendResult(recorded=None, should_close=True)

    assert result.recorded is None
    assert result.should_close
    assert result.closed is None


@pytest.mark.asyncio
async def test_header_delegation_preserves_close_during_request() -> None:
    inner_decision = CloseDuringRequest(after_body_bytes=65536, reset=True)

    async def delegate(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = ctx
        return await call_next()

    async def close(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = (ctx, call_next)
        return inner_decision

    app = compose_headers([delegate, close], _refuse)

    decision = await app(_header_ctx())

    assert decision is inner_decision


@pytest.mark.asyncio
@pytest.mark.parametrize("expected", [True, False])
async def test_header_delegation_passes_bool_decision_through(
    expected: bool,
) -> None:
    async def delegate(
        ctx: HeaderContext, call_next: HeaderNext
    ) -> HeaderDecision:
        _ = ctx
        return await call_next()

    async def terminal(_: HeaderContext) -> HeaderDecision:
        return expected

    app = compose_headers([delegate], terminal)

    decision = await app(_header_ctx())

    assert decision is expected


@pytest.mark.asyncio
async def test_compose_headers_call_next_twice_raises() -> None:
    async def mw(ctx: HeaderContext, call_next: HeaderNext) -> HeaderDecision:
        _ = ctx
        await call_next()
        await call_next()
        raise AssertionError("unreachable")

    app = compose_headers([mw], _allow)

    with pytest.raises(
        RuntimeError, match="call_next\\(\\) called multiple times"
    ):
        await app(_header_ctx())


@pytest.mark.asyncio
async def test_response_sequence_middleware_yields_close_in_order() -> None:
    first = HTTPResponse.text("first")
    close = CloseConnection(reset=True)
    third = HTTPResponse.text("third")
    middleware = ResponseSequenceMiddleware([first, close, third])
    ctx = _responder_ctx(_recorded("GET", "/"))

    results = [await middleware(ctx, _never_next) for _ in range(3)]

    assert results[0] is first
    assert results[1] is close
    assert results[2] is third


@pytest.mark.asyncio
async def test_response_sequence_middleware_delegates_when_exhausted() -> None:
    middleware = ResponseSequenceMiddleware([CloseConnection()])
    ctx = _responder_ctx(_recorded("GET", "/"))
    fallback = HTTPResponse.text("fallback")

    async def call_next(
        *,
        ctx: ResponderContext | None = None,
    ) -> ResponseSpec:
        return fallback

    await middleware(ctx, _never_next)

    assert await middleware(ctx, call_next) is fallback


@pytest.mark.asyncio
async def test_response_sequence_middleware_reset_restarts_sequence() -> None:
    close = CloseConnection()
    middleware = ResponseSequenceMiddleware([close])
    ctx = _responder_ctx(_recorded("GET", "/"))
    await middleware(ctx, _never_next)

    middleware.reset()

    assert await middleware(ctx, _never_next) is close


@pytest.mark.parametrize("times", [0, -1])
def test_close_during_request_factory_rejects_times_below_one(
    times: int,
) -> None:
    with pytest.raises(ValueError, match="times must be at least 1 or None"):
        close_during_request(times=times)


@pytest.mark.asyncio
async def test_close_during_request_times_none_closes_every_request() -> None:
    app = compose_headers(
        [close_during_request(after_body_bytes=64, reset=True)], _refuse
    )

    decisions = [await app(_header_ctx()) for _ in range(3)]

    expected = CloseDuringRequest(after_body_bytes=64, reset=True)
    assert decisions == [expected, expected, expected]


@pytest.mark.asyncio
async def test_close_during_request_times_two_then_delegates() -> None:
    app = compose_headers([close_during_request(times=2)], _allow)

    decisions = [await app(_header_ctx()) for _ in range(3)]

    assert decisions == [CloseDuringRequest(), CloseDuringRequest(), True]


@pytest.mark.asyncio
async def test_throttle_middleware_delegates_while_allowed() -> None:
    middleware = ThrottleMiddleware(_throttler())
    ctx = _responder_ctx(_recorded("GET", "/"))
    allowed = HTTPResponse.text("allowed")

    async def call_next(
        *,
        ctx: ResponderContext | None = None,
    ) -> ResponseSpec:
        return allowed

    assert await middleware(ctx, call_next) is allowed


@pytest.mark.asyncio
async def test_throttle_middleware_denied_request_gets_429() -> None:
    middleware = ThrottleMiddleware(_throttler(rate_per_second=0.5))
    ctx = _responder_ctx(_recorded("GET", "/"))
    await middleware(ctx, _ok_next)

    result = await middleware(ctx, _never_next)

    assert isinstance(result, HTTPResponse)
    assert result.status == 429
    assert result.headers["Retry-After"] == "2"
    assert result.body == b"Too Many Requests"


@pytest.mark.asyncio
async def test_throttle_middleware_uses_custom_response_when_denied() -> None:
    def custom(
        request: RecordedHTTPRequest, decision: ThrottleDecision
    ) -> HTTPResponse:
        body = f"{request.target}:{decision.key}"
        return HTTPResponse.text(body, status=503)

    middleware = ThrottleMiddleware(_throttler(), response=custom)
    ctx = _responder_ctx(_recorded("GET", "/limited"))
    await middleware(ctx, _ok_next)

    result = await middleware(ctx, _never_next)

    assert isinstance(result, HTTPResponse)
    assert result.status == 503
    assert result.body == b"/limited:global"


@pytest.mark.asyncio
async def test_throttle_middleware_reset_refills_the_bucket() -> None:
    middleware = ThrottleMiddleware(_throttler())
    ctx = _responder_ctx(_recorded("GET", "/"))
    await middleware(ctx, _ok_next)

    middleware.reset()

    result = await middleware(ctx, _ok_next)
    assert isinstance(result, HTTPResponse)
    assert result.status == 200


@pytest.mark.asyncio
async def test_raw_proxy_middleware_delegates_non_proxy_request() -> None:
    middleware = RawForwardProxyMiddleware(forwarder=_raw_forwarder())
    ctx = _responder_ctx(_recorded("GET", "/local"))
    local = HTTPResponse.text("local")

    async def call_next(
        *,
        ctx: ResponderContext | None = None,
    ) -> ResponseSpec:
        return local

    assert await middleware(ctx, call_next) is local


@pytest.mark.asyncio
async def test_raw_proxy_middleware_rejects_unparseable_absolute_uri() -> None:
    middleware = RawForwardProxyMiddleware(forwarder=_raw_forwarder())
    ctx = _responder_ctx(_recorded("GET", "http://[::1/broken"))

    result = await middleware(ctx, _never_next)

    assert isinstance(result, HTTPResponse)
    assert result.status == 400
    assert result.body == b"Bad Request: Not an absolute URI"


@pytest.mark.asyncio
async def test_raw_proxy_middleware_rejects_body_rewrite() -> None:
    middleware = RawForwardProxyMiddleware(forwarder=_raw_forwarder())
    recorded = _recorded("POST", "http://example.com/upload", body=b"orig")
    rewritten = replace(
        recorded, request=replace(recorded.request, body=b"changed")
    )

    result = await middleware(_responder_ctx(rewritten), _never_next)

    assert isinstance(result, HTTPResponse)
    assert result.status == 500
    assert result.body == (
        b"Raw proxy forwarding does not support request body rewrites."
    )


@pytest.mark.asyncio
async def test_raw_proxy_middleware_rejects_framing_header_rewrite() -> None:
    middleware = RawForwardProxyMiddleware(forwarder=_raw_forwarder())
    recorded = _recorded("POST", "http://example.com/upload", body=b"data")
    rewritten = recorded.with_headers(
        Headers.from_items([("Content-Length", "4")])
    )

    result = await middleware(_responder_ctx(rewritten), _never_next)

    assert isinstance(result, HTTPResponse)
    assert result.status == 500
    assert result.body == (
        b"Raw proxy forwarding does not support framing header rewrites."
    )


@pytest.mark.asyncio
async def test_raw_proxy_middleware_rejects_connection_framing_token() -> None:
    middleware = RawForwardProxyMiddleware(forwarder=_raw_forwarder())
    recorded = _recorded(
        "GET",
        "http://example.com/",
        headers=Headers.from_items([("Connection", "Transfer-Encoding")]),
    )

    result = await middleware(_responder_ctx(recorded), _never_next)

    assert isinstance(result, HTTPResponse)
    assert result.status == 400
    assert result.body == (
        b"Bad Request: Connection header must not nominate request "
        b"framing: transfer-encoding"
    )


@pytest.mark.asyncio
async def test_raw_proxy_middleware_builds_forward_proxy_response() -> None:
    forwarder = _raw_forwarder()
    middleware = RawForwardProxyMiddleware(forwarder=forwarder)
    recorded = _recorded(
        "GET",
        "https://example.com:8443/path?q=1",
        headers=Headers.from_items([("Host", "example.com")]),
    )

    result = await middleware(_responder_ctx(recorded), _never_next)

    assert isinstance(result, ForwardProxyResponse)
    assert result.host == "example.com"
    assert result.port == 8443
    assert result.upstream_tls
    assert result.request_method == "GET"
    assert result.forwarder is forwarder
    assert result.request_wire_bytes == (
        b"GET /path?q=1 HTTP/1.1\r\nHost: example.com:8443\r\n\r\n"
    )
