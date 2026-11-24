# WebSockets

localstub speaks WebSocket (RFC 6455) so you can test a client's
reconnect logic, its wire format, its message protocol, and its
keepalive handling against a real server. A responder returns
`AcceptWebSocket` wherever it would return an `HTTPResponse`; the
server completes the handshake and gives your test a
`WebSocketSession` to drive.

Because the handshake travels the ordinary responder chain, routes,
`set_response_sequence()`, `set_throttle()`, middleware, and the TLS
intercept proxy all apply to it unchanged.

## Accepting a connection

```python
from localstub import AcceptWebSocket, AsyncHTTPTestServer

async with AsyncHTTPTestServer() as server:
    server.set_default_response(AcceptWebSocket())
    # Point the client at server.ws_url ("ws://127.0.0.1:<port>/").
    ws = await server.next_websocket(timeout=1.0)
    await ws.send_text("hello")
    reply = await ws.receive(timeout=1.0)
    await ws.close()
```

`next_websocket()` returns each session once its 101 is on the wire.
`server.websockets` and `server.last_websocket` hold the sessions
accepted so far. A request that is not a valid handshake gets a 400 or
426 instead, with a body that names the failed check, and is recorded
like any other exchange.

`AcceptWebSocket` options:

| Option | Default | Effect |
| --- | --- | --- |
| `handler` | `None` | Coroutine that runs the session; `None` leaves it for the test |
| `subprotocol` | `None` | `Sec-WebSocket-Protocol` value, sent as given |
| `headers` | `None` | Extra 101 headers; a name here replaces the computed header |
| `compression` | `None` | `PerMessageDeflate` preferences; `None` declines compression |
| `strict` | `True` | Fail the connection when the client violates RFC 6455 |
| `auto_pong` | `True` | Answer each client Ping |
| `auto_close` | `True` | Answer a client Close and end the session |
| `close_timeout` | `5.0` | Seconds `close()` waits for the client's Close |
| `max_message_bytes` | `None` | Largest message accepted, after decompression |

## Driving a session

| Member | Effect |
| --- | --- |
| `await receive(timeout=None)` | Next data message from the client |
| `async for message in ws` | Messages until the session ends |
| `await receive_frame(timeout=None)` | Next client frame, control frames included |
| `await send_text(text)` / `send_bytes(data)` | Send a message; `fragment_size=` splits it, `compress=` overrides compression |
| `await send(message)` | Resend a received `WebSocketMessage`, as an echo does |
| `await ping()` / `pong()` | Send a control frame |
| `await close(code=1000, reason="")` | Closing handshake, then close the transport |
| `await wait_closed(timeout=None)` | The connection's `ConnectionClosed` |
| `auto_pong`, `auto_close` | Flags you can change mid-session |

Receiving and sending raise `WebSocketEnded` once the session has
ended. Messages the client sent before the end are still returned
first.

A handler is an `async def handler(ws: WebSocketSession)`. When it
returns, the server closes with 1000; when it raises, the server
closes with 1011 and records the close reason `error`. A test can take
the session a handler is running from `next_websocket()` as well.

## Faults

Handshake faults use the values localstub already has: an
`HTTPResponse` to reject, `CloseConnection` to close or reset before
answering, and `set_response_sequence()` to fail several attempts and
then accept. `AcceptWebSocket(headers={"Sec-WebSocket-Accept": "bogus"})`
sends a 101 the client must refuse.

During a session:

| Fault | How |
| --- | --- |
| Abnormal closure (1006) | `await ws.drop()` |
| TCP reset | `await ws.drop(reset=True)` |
| Chosen close code | `await ws.close(1013, "try again later")` |
| No pongs | `ws.auto_pong = False` |
| Closing handshake never completes | `ws.auto_close = False` |
| Any frame, valid or not | `await ws.send_frame(WebSocketFrame(...))` |
| Bytes with no framing | `await ws.send_raw(data)` |

`docs/examples/websocket_reconnect.py` rejects a handshake, resets the
next attempt, drops the first session, and checks that the client
resumes on the next one.

## Recording

`session.recorded` is a `RecordedWebSocket`:

- `frames` and `messages` hold both directions in order;
  `client_frames`, `server_frames`, `client_messages`, and
  `server_messages` filter them.
- Each `RecordedWebSocketFrame` has the frame, its exact
  `wire_raw_bytes` (mask key and length encoding included), and
  timestamps.
- `client_close` and `server_close` are the Close frames each side
  sent, and `violations` the protocol rules the client broke.
- `handshake` is the exchange of the upgrade request and the 101.

The histories, and the queues `receive()` and `receive_frame()` read
from, are bounded by `recording_buffer_size` entries and by
`max_connection_bytes` bytes (`0` or `None` for no byte bound). Past
either bound the oldest entries go, counted in `dropped_frames`,
`dropped_messages`, and `session.dropped_unread_messages`; the newest
entry is always kept, however large. The handshake is
an ordinary exchange in `server.exchanges`, and the connection still
ends with one `ConnectionClosed` event in phase `websocket` with
reason `websocket_close`, `websocket_drop`, `client`,
`protocol_error`, `error`, or `shutdown`. Server shutdown closes open
sessions without a Close frame; call `close(1001)` first if the client
should see one.

## Compression

permessage-deflate (RFC 7692) is off unless you ask for it, so recorded
frame payloads equal the application bytes by default:

```python
server.set_default_response(AcceptWebSocket(compression=PerMessageDeflate()))
```

With compression on, each frame records the compressed payload, and
each message records the inflated data with `compressed=True`.

## Through the TLS proxy

`wss://` through `AsyncTLSInterceptProxy` in intercept mode reaches
the server like any HTTPS request, and the sessions behave the same.

In forward mode the proxy tunnels the WebSocket to the real upstream
byte for byte and records both directions.
`proxy.next_websocket()` returns the `RecordedWebSocket`, where
`server` frames are the upstream's. A relayed session can be observed
but not faulted, and its `closed` is always `None`.
