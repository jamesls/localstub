# Changelog

## Unreleased

* Add `max_recorded_body_bytes` (CLI: `--max-recorded-body-bytes`) to keep
  only the first N payload bytes of each request body while still reading
  it to its end, so memory stays bounded for large uploads.
  `RecordedHTTPRequest.full_body_size` gives the size received when a body
  was cut short.
* Add `write_chunk_size` (CLI: `--write-chunk-size`) to send each response
  body as writes of at most N bytes. Over TLS this lowers server CPU for
  large bodies and speeds up fast clients such as the AWS CRT; slower
  readers can see lower throughput, so the default is unchanged.
* Large bodies cost less memory and time: request bodies are freed without
  waiting for the cyclic garbage collector, a recorded request or response
  holds its body once, and request bodies are read with fewer copies and
  larger reads.
* Add connection-lifecycle faults: `CloseConnection` (responder result,
  `reset`, `delay`), `CloseDuringRequest` / `close_during_request()`
  (header-phase close with `after_body_bytes`), `DropConnection(reset=...)`,
  and a keep-alive policy via `set_keep_alive(timeout, max_requests, reset,
  advertise)` or the `keep_alive_timeout` / `max_requests_per_connection`
  constructor arguments. Time is injectable through the `sleep` argument.
* Record exactly one `ConnectionClosed` event per connection (reason, phase,
  reset, requests completed, byte counters, timestamp), exposed through
  `closed_connections`, `last_closed_connection`, `next_closed_connection()`,
  and `dropped_closed_connections` on both `AsyncHTTPTestServer` and
  `AsyncTLSInterceptProxy`. `RecordedExchange` gains `interim_responses`
  and `closed`.
* Add `close_http_connection(writer)` for handed-off connections; the TLS
  intercept proxy uses it so `aclose()` records `shutdown` for every
  intercepted connection.
* CLI: `{"type": "close", "reset": ..., "delay": ...}` responses in config
  files, plus `--keep-alive-timeout` and `--max-requests-per-connection`.
* `AsyncioClient` pools upstream connections per origin and reuses them
  across `send()` calls (`max_connections_per_origin=10`,
  `max_idle_connections=20`, `idle_timeout=5.0`). It is exported from
  `localstub` and closes its pool with `aclose()` or `async with`.
  `AsyncHTTPTestServer(forward_proxy=True)` builds a server-owned client
  and closes it in `aclose()`; the CLI uses this.
* Add `AsyncTLSInterceptProxy.client_env_vars()`, which returns the environment
  variables that route a client's HTTPS traffic through the proxy:
  `HTTPS_PROXY`, plus `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`,
  `CURL_CA_BUNDLE` and `AWS_CA_BUNDLE` pointing at the proxy's CA. The
  `lstub` CLI prints them as `export` lines when it starts.
* Header values may contain non-ASCII octets (obs-text). `Headers` keeps
  the original bytes and decodes them as Latin-1; `Headers.raw` exposes the
  bytes and `Headers.from_raw_items()` builds headers from them. Server
  responses, the asyncio client and the forward proxy write values back as
  Latin-1 instead of failing or substituting replacement characters.
* Fixes:
  * `server.url` brackets IPv6 hosts (`http://[::1]:8080/`).
  * Proxy requests are recognized regardless of scheme case, so
    `HTTP://...` is forwarded instead of receiving the stub response.
  * Absolute-form URLs keep `;` path parameters, which were previously
    dropped before route matching and forwarding.
  * A response that sets `Transfer-Encoding` no longer also gets a
    computed `Content-Length`.
  * An upstream that never answers TLS `close_notify` no longer delays
    the next forwarded request on a keep-alive connection.
  * `AsyncTLSInterceptProxy.aclose()` no longer hangs when a client sends
    `CONNECT` but never starts TLS, or its TLS handshake times out.
* Compatibility notes:
  * `SendResult.recorded` is now optional and `SendResult` gains `closed`;
    sender middleware must check for `None` before reading response fields.
  * `HeaderNext` returns `Awaitable[HeaderDecision]`
    (`bool | CloseDuringRequest`). Delegating middleware must return the
    decision unchanged; coercing `CloseDuringRequest` to `bool` raises
    `TypeError`.
  * Header-phase rejection (`False`) now records the partial request, the
    early response, and a `request_read` close event instead of nothing.
    A final header-phase send followed by `True` drains the body and skips
    the responder and sender chains.
  * Requests whose body was not read to the end are now recorded, with
    `RecordedHTTPRequest.body_complete` set to `False`: interrupted
    uploads, body parse errors, header-phase rejections, and shutdown
    mid-body. They appear in `requests`, `last_request`, `next_request()`
    and `exchanges`, so request counts can grow; filter on
    `body_complete` to keep only full requests.
  * Shutting down `AsyncTLSInterceptProxy` (including SIGTERM to the CLI)
    cuts in-flight requests instead of letting them finish. The client
    sees EOF, and the partial exchange is recorded with a `shutdown`
    close reason.
  * Pooled `AsyncioClient` connections stay open until `aclose()`. The
    server never closes an injected client, so an inline
    `upstream_client=AsyncioClient()` leaks its sockets and triggers
    `ResourceWarning` at loop shutdown; use `forward_proxy=True` or
    `async with AsyncioClient()`. Passing both `forward_proxy=True` and
    `upstream_client=` raises `ValueError`. Upstreams now see reused
    connections and no longer receive `Connection: close`.
  * The `HTTPClient` protocol gains `aclose()`. The server never calls
    it on injected clients, so this only affects type checking.
  * `TrafficRecorder.record_exchange()` takes `interim_responses` and
    `closed` keyword arguments, and the server also calls the new
    `record_connection_closed()`. Subclasses that override
    `record_exchange()` must accept the new arguments.
  * `Headers.from_items()` and `patch_set()` raise `ValueError` for a
    non-ASCII name or a value outside the Latin-1 range. Incoming headers
    are still decoded as Latin-1.
  * `AsyncRequestParser.parse()`, `parse_headers()` and
    `continue_parse_body()` return `ParseOutcome`; only
    `ParseOutcome.complete_request` is a full request.
  * Custom `TransmissionStrategy` subclasses: `write_body` may return
    `AbortTransmission`, and the `Writer` protocol is reduced to `write`
    and `drain`. The writer passed in no longer has `writelines`,
    `write_eof`, `is_closing` or `get_extra_info`. A strategy that used to
    close the writer must return `AbortTransmission` instead.
  * `default_response` returns `HTTPResponse | CloseConnection`, and
    `set_response_sequence()` accepts `CloseConnection` items.
  * Traffic JSONL adds `body_complete` to requests and `interim_responses`
    and `closed` (or `null`) to exchanges. Partial requests also produce
    records, often with `"response": null`.
  * `localstub.server` is now a package (`localstub.server.core`,
    `localstub.server.connection`, `localstub.server.transmission`); the
    public names are still importable from `localstub.server`. Names it
    only imported from elsewhere (such as `Router`, `TrafficRecorder`,
    `HTTPClient`, `ConnectionMeta` and `MonotonicClock`) are not; import
    them from their defining modules.
  * `RouterMiddleware` and `HandlerMiddleware` moved from
    `localstub.middleware.builtins` to `localstub.router`,
    `upstream_ssl_context` moved from `localstub.forward` to
    `localstub.http.upstream`, and `localstub.http.framing.chunk_payloads`
    was removed.

## v0.0.3

* Add per-connection cap of max bytes recorded (#13)


## v0.0.2

* Support persistent connections in forward mode (#12)


## v0.0.1

* Initial release
