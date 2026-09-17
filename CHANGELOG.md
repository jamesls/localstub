# Changelog

## Unreleased

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
  * `RecordedHTTPRequest.body_complete` is `False` for interrupted uploads
    and body parse errors; request consumers must handle partial records.
  * `AsyncRequestParser.parse()`, `parse_headers()` and
    `continue_parse_body()` return `ParseOutcome`; only
    `ParseOutcome.complete_request` is a full request.
  * Custom `TransmissionStrategy` subclasses: `write_body` may return
    `AbortTransmission`, and the `Writer` protocol loses `close` and
    `wait_closed`. A strategy that used to close the writer must return
    `AbortTransmission` instead.
  * `default_response` returns `HTTPResponse | CloseConnection`, and
    `set_response_sequence()` accepts `CloseConnection` items.
  * Traffic JSONL adds `body_complete` to requests and `interim_responses`
    and `closed` (or `null`) to exchanges.
  * `localstub.server` is now a package (`localstub.server.core`,
    `localstub.server.connection`, `localstub.server.transmission`); the
    public names are still importable from `localstub.server`.

## v0.0.3

* Add per-connection cap of max bytes recorded (#13)


## v0.0.2

* Support persistent connections in forward mode (#12)


## v0.0.1

* Initial release
