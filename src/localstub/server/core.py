from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from types import TracebackType
from typing import Any, Self

from localstub.forward import RawForwarder
from localstub.http.client import HTTPClient
from localstub.http.clients.asyncio import AsyncioClient
from localstub.http.exchange import ConnectionClosed, RecordedExchange
from localstub.http.request import (
    HTTPRequestHeaders,
    RecordedHTTPRequest,
    client_address,
)
from localstub.http.response import RecordedHTTPResponse
from localstub.http.responsespec import HeadersLike, HTTPResponse
from localstub.http.utils import maybe_await
from localstub.middleware import (
    CloseConnection,
    HeaderContext,
    HeaderDecision,
    HeaderMiddleware,
    HeaderNext,
    ResponderContext,
    ResponderMiddleware,
    ResponseSpec,
    SenderMiddleware,
    ServerServices,
    SystemTimestampProvider,
    TimestampProvider,
    compose_headers,
    compose_responder,
    compose_sender,
)
from localstub.middleware.builtins import (
    BuiltinMiddlewares,
    ForwardProxyMiddleware,
    RawForwardProxyMiddleware,
    ResponseSequenceMiddleware,
    ThrottleMiddleware,
    ThrottleResponseFunc,
    default_throttle_response,
)
from localstub.recording import (
    DEFAULT_MAX_CONNECTION_BYTES,
    DEFAULT_RECORDING_BUFFER_SIZE,
    BoundedByteBuffer,
    TrafficRecorder,
)
from localstub.router import (
    HandlerMiddleware,
    ResponderHandler,
    Router,
    RouterMiddleware,
)
from localstub.server.connection import (
    CaptureContext,
    ConnectionState,
    CountingStreamReader,
    HeaderApp,
    HTTPConnection,
    KeepAlivePolicy,
    RecordingStreamWriter,
    RequestPipeline,
    ResponderApp,
    SenderApp,
    Sleep,
)
from localstub.server.transmission import (
    ImmediateTransmission,
    TransmissionStrategy,
)
from localstub.throttle import (
    Clock,
    MonotonicClock,
    ThrottleDecision,
    ThrottleKeyFunc,
    TokenBucketThrottler,
)

LOG = logging.getLogger(__name__)
ThrottleResponse = HTTPResponse | ThrottleResponseFunc

# Event-loop turns aclose() yields after pausing accepts, so already-accepted
# connections reach _client_connected and get torn down. Turning an accepted
# socket into that callback took up to 5 turns when measured on CPython 3.12
# (worst case: a connection made by a blocking connect()); this leaves
# headroom above that. These are bare sleep(0) yields, so unused turns cost
# nothing beyond a few trips through the loop.
_SHUTDOWN_DRAIN_TURNS = 8


def _pause_server_accepts(server: asyncio.Server) -> None:
    loop = asyncio.get_running_loop()
    try:
        for listener in server.sockets:
            loop.remove_reader(listener.fileno())
    except NotImplementedError:
        # Proactor loops attach accepted transports in the accept callback, so
        # they have no selector task that can be stranded by Server.close().
        server.close()


async def _drain_pending_accepts() -> None:
    for _ in range(_SHUTDOWN_DRAIN_TURNS):
        await asyncio.sleep(0)


def _default_throttle_key(_: RecordedHTTPRequest) -> str:
    return "global"


# Callback to send a response to client during header processing
SendResponse = Callable[[HTTPResponse], Awaitable[None]]

# Lifecycle hook called after headers are received, before body is read.
# Return True to continue reading body, False to stop reading and close
# the connection; the partial request and the close are recorded.
OnHeadersReceived = Callable[
    [HTTPRequestHeaders, SendResponse],
    Awaitable[bool] | bool,
]


class AsyncHTTPTestServer:
    """Small asyncio HTTP server used for testing SDK clients.

    Features:
      * exposes .url (e.g. "http://127.0.0.1:12345/")
      * records last_request (RecordedHTTPRequest) and a list of all
        requests
      * `wire_raw_bytes` contains the *exact* bytes received, including
        chunked framing and trailers.
      * configurable static response, or plug in responder middleware.
      * connection-lifecycle faults: ``CloseConnection`` in place of a
        response, ``CloseDuringRequest`` from header middleware, and a
        keep-alive policy (``set_keep_alive``); every connection ends
        with one ``ConnectionClosed`` event in ``closed_connections``.
      * HTTP forward proxying: ``forward_proxy=True`` gives the server
        its own ``AsyncioClient``, closed when the server closes.
        ``upstream_client=`` injects a client you constructed; the
        server never closes an injected client.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        handler: ResponderHandler | None = None,
        default_response: HTTPResponse | CloseConnection | None = None,
        on_headers_received: OnHeadersReceived | None = None,
        upstream_client: HTTPClient | None = None,
        raw_forwarder: RawForwarder | None = None,
        clock: Clock | None = None,
        timestamp_provider: TimestampProvider | None = None,
        recording_buffer_size: int | None = DEFAULT_RECORDING_BUFFER_SIZE,
        recorder: TrafficRecorder | None = None,
        *,
        forward_proxy: bool = False,
        max_connection_bytes: int | None = DEFAULT_MAX_CONNECTION_BYTES,
        keep_alive_timeout: float | None = None,
        max_requests_per_connection: int | None = None,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        if max_connection_bytes is not None and max_connection_bytes < 0:
            raise ValueError(
                "max_connection_bytes must be non-negative or None, "
                f"got {max_connection_bytes}"
            )
        if forward_proxy and upstream_client is not None:
            raise ValueError(
                "pass either forward_proxy=True or upstream_client, not both"
            )
        self._host = host
        self._port = port
        self._server: asyncio.base_events.Server | None = None
        self._closing = False
        self._connections: dict[asyncio.StreamWriter, HTTPConnection] = {}
        self._client_tasks: dict[HTTPConnection, asyncio.Task[None]] = {}

        self._handler: ResponderHandler | None = handler
        self._default_response: HTTPResponse | CloseConnection = (
            default_response or HTTPResponse.json({})
        )
        self.router = Router()
        self._on_headers_received = on_headers_received
        self._owned_upstream_client_factory: (
            Callable[[], HTTPClient] | None
        ) = None
        if forward_proxy:
            self._owned_upstream_client_factory = AsyncioClient
            upstream_client = self._owned_upstream_client_factory()
        self._upstream_client = upstream_client
        self._raw_forwarder = raw_forwarder

        self._builtins = BuiltinMiddlewares()
        if raw_forwarder is not None:
            self._builtins.set(
                "raw_proxy", RawForwardProxyMiddleware(raw_forwarder)
            )
        if upstream_client is not None:
            self._builtins.set(
                "proxy", ForwardProxyMiddleware(upstream_client)
            )

        self.responder_middlewares: list[ResponderMiddleware] = []
        self.sender_middlewares: list[SenderMiddleware] = []
        self.header_middlewares: list[HeaderMiddleware] = []

        self._transmission_strategy: TransmissionStrategy = (
            ImmediateTransmission()
        )
        self._keep_alive = KeepAlivePolicy(
            timeout=keep_alive_timeout,
            max_requests=max_requests_per_connection,
        )
        self._sleep = sleep

        self._clock: Clock = clock or MonotonicClock()
        self._timestamp_provider = (
            timestamp_provider or SystemTimestampProvider()
        )
        self._services = ServerServices(
            clock=self._clock,
            timestamp_provider=self._timestamp_provider,
        )

        # Recorded history and queues are bounded so memory stays flat
        # when the server runs long enough that traffic outpaces whatever
        # is consuming the records (e.g. the CLI's forward-proxy mode).
        self._recorder = recorder or TrafficRecorder(
            recording_buffer_size,
            clock=self._clock,
            timestamp_provider=self._timestamp_provider,
        )
        self._max_connection_bytes = max_connection_bytes

        # Connection-level raw bytes retained by client address, so tests
        # can read them after the connection is gone.
        self._connection_raw_bytes_received: dict[
            tuple[str, int], BoundedByteBuffer
        ] = {}
        self._connection_raw_bytes_sent: dict[
            tuple[str, int], BoundedByteBuffer
        ] = {}

        self._pipeline = RequestPipeline(
            header=self._build_header,
            responder=self._build_responder,
            sender=self._build_sender,
            transmission=self._get_transmission_strategy,
            keep_alive=self._get_keep_alive,
        )

        self.host: str | None = None
        self.port: int | None = None

    @property
    def url(self) -> str:
        if self.host is None or self.port is None:
            raise RuntimeError("Server not started yet")
        return f"http://{self.host}:{self.port}/"

    @property
    def handler(self) -> ResponderHandler | None:
        return self._handler

    @handler.setter
    def handler(self, value: ResponderHandler | None) -> None:
        self._handler = value

    @property
    def default_response(self) -> HTTPResponse | CloseConnection:
        return self._default_response

    @default_response.setter
    def default_response(
        self, response: HTTPResponse | CloseConnection
    ) -> None:
        self.set_default_response(response)

    def add_route(
        self,
        method: str,
        path: str,
        handler: ResponderHandler,
    ) -> None:
        self.router.add(method, path, handler)

    def use(self, middleware: ResponderMiddleware) -> None:
        self.responder_middlewares.append(middleware)

    def use_sender(self, middleware: SenderMiddleware) -> None:
        self.sender_middlewares.append(middleware)

    def use_headers(self, middleware: HeaderMiddleware) -> None:
        self.header_middlewares.append(middleware)

    def set_request_headers_handler(
        self,
        handler: OnHeadersReceived,
    ) -> None:
        """Set handler when client request headers are received.

        This will overwrite the `on_headers_received` value if one was
        provided when this class was instantiated.  Returning ``False``
        stops reading the request and closes the connection, recording
        the partial request and a ``request_read`` close event, the
        same as header middleware returning ``False``.

        """
        self._on_headers_received = handler

    def set_json_response(
        self,
        obj: Any,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
    ) -> None:
        """Configure a static JSON response returned for every request."""
        self.set_default_response(
            HTTPResponse.json(obj, status=status, headers=headers)
        )

    def set_text_response(
        self,
        text: str,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
    ) -> None:
        self.set_default_response(
            HTTPResponse.text(text, status=status, headers=headers)
        )

    def set_raw_response(
        self,
        data: bytes,
        *,
        status: int = 200,
        headers: HeadersLike | None = None,
    ) -> None:
        self.set_default_response(
            HTTPResponse.raw(data, status=status, headers=headers)
        )

    def set_default_response(
        self, response: HTTPResponse | CloseConnection
    ) -> None:
        """Configure a static response returned for every request.

        Unlike set_json_response/set_text_response/set_raw_response, this
        accepts an already-constructed HTTPResponse object, or a
        ``CloseConnection`` to close on every request without a handler.

        Args:
            response: HTTPResponse or CloseConnection used for all requests
        """
        self._default_response = response
        self._builtins.clear("sequence")

    def set_response_sequence(
        self, responses: list[HTTPResponse | CloseConnection]
    ) -> None:
        """Configure a sequence of responses to return in order.

        Each incoming request will consume the next response from the sequence.
        Once exhausted, falls back to handler or default_response behavior.

        This is useful for testing retry logic where you want the first N
        requests to fail and subsequent requests to succeed.  An item may
        be a ``CloseConnection`` to close instead of responding.

        Example:
            server.set_response_sequence([
                HTTPResponse(status=500),  # First request fails
                CloseConnection(),  # Second request gets no response
                HTTPResponse.json({"ok": True})  # Third request succeeds
            ])

        Args:
            responses: List of HTTPResponse or CloseConnection objects to
                return in sequence
        """
        self._builtins.set("sequence", ResponseSequenceMiddleware(responses))
        # Clear default response (last one wins)
        self._default_response = HTTPResponse.json({})

    def set_transmission_strategy(
        self, strategy: TransmissionStrategy
    ) -> None:
        """Configure how response body bytes are transmitted.

        This controls the network transmission behavior (e.g., throttling,
        chunking) without changing the actual response content. Useful for
        testing client behavior under various network conditions.

        Example:
            # Simulate slow S3 GetObject response
            server.set_raw_response(large_file_bytes)
            server.set_transmission_strategy(
                ThrottledTransmission(chunk_size=8192, delay=0.1)
            )

        Args:
            strategy: TransmissionStrategy instance controlling transmission
        """
        self._transmission_strategy = strategy

    def set_keep_alive(
        self,
        *,
        timeout: float | None = None,
        max_requests: int | None = None,
        reset: bool = False,
        advertise: bool = False,
    ) -> None:
        """Configure how connections are kept alive between requests.

        Each call replaces all four settings, so ``set_keep_alive()``
        restores unlimited keep-alive without advertised hints.  Open
        connections see the new policy on their next request.

        Args:
            timeout: Seconds to wait after a response for the next
                request's first byte before closing with an
                ``idle_timeout`` event.  ``None`` waits forever; ``0.0``
                closes right after every response.
            max_requests: Close after this many completed final
                responses on one connection with a ``max_requests``
                event, adding ``Connection: close`` to the last local
                response.
            reset: Make timeout closures abortive TCP closes.
            advertise: Add ``Keep-Alive: timeout=N, max=M`` hints to
                locally constructed responses that allow reuse.
        """
        self._keep_alive = KeepAlivePolicy(
            timeout=timeout,
            max_requests=max_requests,
            reset=reset,
            advertise=advertise,
        )

    def get_connection_bytes_received(
        self, client: tuple[str, int]
    ) -> bytes | None:
        """Get retained raw bytes received from a client connection.

        Args:
            client: Tuple of (host, port) identifying the client connection

        Returns:
            Retained bytes from this client, or None if no data recorded
        """
        buf = self._connection_raw_bytes_received.get(client)
        return bytes(buf) if buf is not None else None

    def get_connection_bytes_sent(
        self, client: tuple[str, int]
    ) -> bytes | None:
        """Get retained raw bytes sent to a client connection.

        Args:
            client: Tuple of (host, port) identifying the client connection

        Returns:
            Retained bytes sent to this client, or None if no data recorded
        """
        buf = self._connection_raw_bytes_sent.get(client)
        return bytes(buf) if buf is not None else None

    def get_connection_dropped_bytes_received(
        self, client: tuple[str, int]
    ) -> int | None:
        """Get the count of received bytes dropped for a connection."""
        buf = self._connection_raw_bytes_received.get(client)
        return buf.dropped if buf is not None else None

    def get_connection_dropped_bytes_sent(
        self, client: tuple[str, int]
    ) -> int | None:
        """Get the count of sent bytes dropped for a connection."""
        buf = self._connection_raw_bytes_sent.get(client)
        return buf.dropped if buf is not None else None

    def get_request_timestamp(self, request: RecordedHTTPRequest) -> float:
        """Get the reception timestamp for a request.

        Args:
            request: The RecordedHTTPRequest object to look up.

        Returns:
            Monotonic timestamp when the request was received.

        Raises:
            ValueError: If the request is not found.  Timestamps are only
                retained for requests still in the ``requests`` history,
                which is bounded by ``recording_buffer_size``.
        """
        return self._recorder.get_request_timestamp(request)

    def clear_requests(self) -> None:
        """Clear all recorded request state.

        Resets last_request, requests list, the request queue, the
        closed-connection history and queue, and connection-level raw
        bytes while preserving server configuration (handler,
        default_response, etc.).  Per-connection byte counters are not
        touched.

        Useful for reusing a session-scoped test server across multiple
        tests without needing to shut down and restart the server.

        Example:
            async with AsyncHTTPTestServer() as server:
                # Test 1
                response = await client.get(server.url)
                assert len(server.requests) == 1

                # Clear state between tests
                server.clear_requests()

                # Test 2 - fresh state
                response = await client.get(server.url)
                assert len(server.requests) == 1
        """
        self._recorder.reset()
        for buf in self._connection_raw_bytes_received.values():
            buf.clear()
        for buf in self._connection_raw_bytes_sent.values():
            buf.clear()
        self._connection_raw_bytes_received.clear()
        self._connection_raw_bytes_sent.clear()
        self._builtins.reset_all()

    def set_throttle(
        self,
        *,
        rate_per_second: float,
        key: ThrottleKeyFunc | None = None,
        burst: float | None = None,
        response: ThrottleResponse | None = None,
        clock: Clock | None = None,
    ) -> None:
        """Enable request-rate throttling for this server.

        This is request-per-second throttling enforced via a token bucket.
        When throttled, the server returns the configured throttling response.

        Args:
            rate_per_second: Token bucket refill rate (requests per second).
            key: Function mapping a request to a hashable throttle key.
            burst: Max burst capacity for each key bucket (must be >= 1).
            response: Static HTTPResponse or callable response builder used
                when a request is throttled.
            clock: Optional clock for deterministic testing.
        """
        key_fn = key or _default_throttle_key
        throttler = TokenBucketThrottler(
            rate_per_second=rate_per_second,
            key=key_fn,
            burst=burst,
            clock=clock,
        )
        response_fn = self._normalize_throttle_response(response)
        self._builtins.set(
            "throttle",
            ThrottleMiddleware(throttler=throttler, response=response_fn),
        )

    def clear_throttle(self) -> None:
        """Disable request-rate throttling."""
        self._builtins.clear("throttle")

    def _normalize_throttle_response(
        self,
        response: ThrottleResponse | None,
    ) -> ThrottleResponseFunc:
        if response is None:
            return default_throttle_response
        if isinstance(response, HTTPResponse):

            def static(
                _: RecordedHTTPRequest,
                __: ThrottleDecision,
                *,
                _response: HTTPResponse = response,
            ) -> HTTPResponse:
                return _response

            return static
        return response

    async def start(self) -> None:
        if self._server is not None:
            return

        self._ensure_owned_upstream_client()
        self._closing = False
        self._server = await asyncio.start_server(
            self._client_connected,
            self._host,
            self._port,
        )
        assert self._server.sockets
        sockname = self._server.sockets[0].getsockname()
        self.host, self.port = sockname[0], sockname[1]

    async def aclose(self) -> None:
        """Stop the listener and shut down every tracked connection.

        Every connection, including streams handed to
        ``handle_http_connection``, records a ``shutdown`` close unless
        its close was already decided.  Listener shutdown and
        connection cleanup are independent, so handed-off connections
        are shut down even when ``start()`` was never called.

        Called from inside a handler, this returns while that handler's
        own connection is still open; it closes once the handler's
        response has been sent.
        """
        server = self._server
        if server is None:
            self._shutdown_connections()
            await self._close_owned_upstream_client()
            return
        self._closing = True
        cancelled: asyncio.CancelledError | None = None
        try:
            # Stop selector loops from accepting more sockets without closing
            # the asyncio.Server yet. Closing it while an already-accepted
            # socket is still attaching its transport can strand the peer.
            _pause_server_accepts(server)
            drain_task = asyncio.create_task(_drain_pending_accepts())
            try:
                await asyncio.shield(drain_task)
            except asyncio.CancelledError as exc:
                # Finish the critical drain before honoring cancellation.
                await drain_task
                cancelled = exc

            server.close()
            self._shutdown_connections()
            caller = self._calling_connection()
            client_tasks = tuple(
                task
                for connection, task in self._client_tasks.items()
                if connection is not caller
            )
            for task in client_tasks:
                task.cancel()
            if client_tasks:
                await asyncio.gather(*client_tasks, return_exceptions=True)
            if caller is None:
                # Server.wait_closed() waits for every accepted transport,
                # and the caller's stays open until its handler returns.
                await server.wait_closed()
        finally:
            server.close()
            if self._server is server:
                self._server = None
            self._closing = False
            await self._close_owned_upstream_client()

        if cancelled is not None:
            raise cancelled

    def close_http_connection(self, writer: asyncio.StreamWriter) -> None:
        """Decide shutdown for a tracked connection and initiate closure.

        Intended for callers such as the TLS proxy that own the streams
        passed to ``handle_http_connection``.  An earlier close decision
        is preserved, and an untracked or finished connection is a
        no-op.  The loop is interrupted so it finishes cleanup and
        publishes its event without waiting for a delay, an idle timer,
        or the client.  Other connections are not affected.
        """
        connection = self._connections.get(writer)
        if connection is not None:
            connection.shutdown()

    def _shutdown_connections(self) -> None:
        for connection in tuple(self._connections.values()):
            connection.shutdown()

    def _calling_connection(self) -> HTTPConnection | None:
        """The accepted connection whose loop is running the caller.

        ``aclose()`` may run inside a handler.  That connection cannot
        finish until ``aclose()`` returns, so shutdown neither cancels
        its task nor waits for its transport; the decision is recorded
        and the loop exits after the current request.
        """
        for connection in self._client_tasks:
            if connection.owns_current_task():
                return connection
        return None

    def _ensure_owned_upstream_client(self) -> None:
        factory = self._owned_upstream_client_factory
        if factory is None or self._upstream_client is not None:
            return
        client = factory()
        self._upstream_client = client
        self._builtins.set("proxy", ForwardProxyMiddleware(client))

    async def _close_owned_upstream_client(self) -> None:
        if self._owned_upstream_client_factory is None:
            return
        client = self._upstream_client
        if client is None:
            return
        try:
            await client.aclose()
        finally:
            if self._upstream_client is client:
                self._upstream_client = None
                self._builtins.clear("proxy")

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    @property
    def recorder(self) -> TrafficRecorder:
        """The recorder shared with proxies attached to this server."""
        return self._recorder

    @property
    def requests(self) -> list[RecordedHTTPRequest]:
        """All recorded requests, oldest first (bounded history)."""
        return self._recorder.requests

    @property
    def last_request(self) -> RecordedHTTPRequest | None:
        """The most recently recorded or consumed request."""
        return self._recorder.last_request

    @property
    def responses(self) -> list[RecordedHTTPResponse]:
        """All recorded responses, oldest first (bounded history)."""
        return self._recorder.responses

    @property
    def last_response(self) -> RecordedHTTPResponse | None:
        """The most recently recorded or consumed response."""
        return self._recorder.last_response

    @property
    def exchanges(self) -> list[RecordedExchange]:
        """All recorded exchanges, oldest first (bounded history)."""
        return self._recorder.exchanges

    @property
    def last_exchange(self) -> RecordedExchange | None:
        """The most recently recorded exchange."""
        return self._recorder.last_exchange

    @property
    def closed_connections(self) -> list[ConnectionClosed]:
        """Close events of finished connections, oldest first (bounded)."""
        return self._recorder.closed_connections

    @property
    def last_closed_connection(self) -> ConnectionClosed | None:
        """The most recently recorded close event."""
        return self._recorder.last_closed_connection

    async def next_request(
        self, timeout: float | None = None
    ) -> RecordedHTTPRequest:
        """Await and return the next request that hits this server."""
        return await self._recorder.next_request(timeout)

    async def next_response(
        self, timeout: float | None = None
    ) -> RecordedHTTPResponse:
        """Await and return the next response sent by this server."""
        return await self._recorder.next_response(timeout)

    async def next_exchange(
        self, timeout: float | None = None
    ) -> RecordedExchange:
        """Await and return the next completed request/response exchange."""
        return await self._recorder.next_exchange(timeout)

    def next_exchange_nowait(self) -> RecordedExchange | None:
        """Return the next completed exchange, or None if none is queued."""
        return self._recorder.next_exchange_nowait()

    async def next_closed_connection(
        self, timeout: float | None = None
    ) -> ConnectionClosed:
        """Await and return the next connection close event.

        Events are published when a connection is finalized, so the
        queue order is finalization order, not decision order.  When
        connections overlap, correlate through ``exchange.closed``.
        """
        return await self._recorder.next_closed_connection(timeout)

    @property
    def dropped_requests(self) -> int:
        """Requests evicted unread from the next_request() buffer."""
        return self._recorder.dropped_requests

    @property
    def dropped_responses(self) -> int:
        """Responses evicted unread from the next_response() buffer."""
        return self._recorder.dropped_responses

    @property
    def dropped_exchanges(self) -> int:
        """Exchanges evicted unread from the next_exchange() buffer."""
        return self._recorder.dropped_exchanges

    @property
    def dropped_closed_connections(self) -> int:
        """Close events evicted unread from the next_closed_connection()
        buffer."""
        return self._recorder.dropped_closed_connections

    async def handle_http_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle an HTTP conversation on externally-established streams.

        This method handles the full HTTP request/response cycle on streams
        that may have been established externally (e.g., by a TLS proxy
        that terminates TLS and hands off the decrypted streams).

        The server will:
        1. Parse incoming HTTP requests
        2. Record requests in .requests and .last_request
        3. Generate responses using the configured handler/routes
        4. Write responses to the client
        5. Handle connection persistence (keep-alive vs close)
        6. Record one ConnectionClosed event when the connection ends

        Args:
            reader: StreamReader for the client connection
            writer: StreamWriter for the client connection

        Note:
            This method is intended for external callers like TLS proxies.
            The server manages cleanup of the writer on exit.  Nothing
            may read from ``reader`` directly once it is handed off.
            Read-ahead left on ``reader`` by an earlier parse through
            ``localstub.http.stream`` is served before new stream data,
            so a pipelined request read past before the handoff is
            still served and recorded.  ``close_http_connection(writer)``
            shuts the conversation down early.
        """
        connection = self._open_connection(reader, writer)
        if self._closing:
            connection.shutdown()
        await self._run_connection(connection, writer)

    async def _run_connection(
        self,
        connection: HTTPConnection,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            await connection.run()
        finally:
            if self._connections.get(writer) is connection:
                del self._connections[writer]

    def _open_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> HTTPConnection:
        client = client_address(writer)
        received, sent = self._new_connection_buffers(client)
        state = ConnectionState(client=client, received=received, sent=sent)
        connection = HTTPConnection(
            reader=CountingStreamReader(reader, state),
            writer=RecordingStreamWriter(writer, sent, state=state),
            state=state,
            pipeline=self._pipeline,
            recorder=self._recorder,
            services=self._services,
            sleep=self._sleep,
        )
        self._connections[writer] = connection
        return connection

    def _new_connection_buffers(
        self, client: tuple[str, int] | None
    ) -> tuple[BoundedByteBuffer | None, BoundedByteBuffer | None]:
        """Create or reuse the retained byte buffers for a client."""
        if client is None or self._max_connection_bytes == 0:
            return None, None

        if client not in self._connection_raw_bytes_received:
            self._connection_raw_bytes_received[client] = BoundedByteBuffer(
                self._max_connection_bytes
            )
        if client not in self._connection_raw_bytes_sent:
            self._connection_raw_bytes_sent[client] = BoundedByteBuffer(
                self._max_connection_bytes
            )

        return (
            self._connection_raw_bytes_received[client],
            self._connection_raw_bytes_sent[client],
        )

    def _client_connected(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        connection = self._open_connection(reader, writer)
        if self._closing:
            # Accepted so late in shutdown that aclose() already snapshotted
            # the tasks to wait for. Finalize right away so the client still
            # gets a FIN and the connection still records its event.
            connection.shutdown()
            connection.finalize()
            del self._connections[writer]
            return
        task = asyncio.create_task(self._run_connection(connection, writer))
        self._client_tasks[connection] = task

        def _release_client(done_task: asyncio.Task[None]) -> None:
            _ = done_task
            self._client_tasks.pop(connection, None)

        task.add_done_callback(_release_client)

    def _get_transmission_strategy(self) -> TransmissionStrategy:
        return self._transmission_strategy

    def _get_keep_alive(self) -> KeepAlivePolicy:
        return self._keep_alive

    def _uses_static_response(self) -> bool:
        return (
            not self.responder_middlewares
            and not self._builtins.any_active
            and not self.router.has_routes
            and self._handler is None
        )

    def _build_responder(
        self,
        capture_ctx: CaptureContext | None,
    ) -> ResponderApp:
        default_response = self._default_response
        if self._uses_static_response():

            async def static(_: ResponderContext) -> ResponseSpec:
                return default_response

            return static

        middlewares: list[ResponderMiddleware] = [
            *self.responder_middlewares,
            *self._builtins.active(),
            RouterMiddleware(self.router),
            HandlerMiddleware(lambda: self._handler),
        ]

        def terminal(_: ResponderContext) -> ResponseSpec:
            return default_response

        return compose_responder(
            middlewares,
            terminal,
            capture_ctx=capture_ctx,
        )

    def _build_sender(self, terminal: SenderApp) -> SenderApp | None:
        if not self.sender_middlewares:
            return None
        return compose_sender([*self.sender_middlewares], terminal)

    def _build_header(self) -> HeaderApp | None:
        middlewares: list[HeaderMiddleware] = []
        on_headers_received_handler = self._on_headers_received
        if on_headers_received_handler is not None:

            async def on_headers_received(
                ctx: HeaderContext,
                call_next: HeaderNext,
            ) -> HeaderDecision:
                should_continue: bool = await maybe_await(
                    on_headers_received_handler(ctx.headers, ctx.send)
                )
                if not should_continue:
                    return False
                return await call_next()

            middlewares.append(on_headers_received)

        middlewares.extend(self.header_middlewares)
        if not middlewares:
            return None

        async def terminal(_: HeaderContext) -> HeaderDecision:
            return True

        return compose_headers(middlewares, terminal)
