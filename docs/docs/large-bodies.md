# Large bodies and benchmarking

localstub records every request it receives, so a test can assert on
it afterwards. With large uploads, such as multi-gigabyte S3 objects,
or when the server is the endpoint of a throughput benchmark, those
recordings are what uses memory. This page shows the settings that
keep memory bounded.

## What the server keeps

- Each recorded request holds its body. For a `Content-Length` body,
  `wire_raw_bytes` shares it and is assembled only when you read it,
  so the body is held once. A chunked body's wire bytes include the
  chunk framing, so they are held alongside the body.
- Each recorded response shares the body of the `HTTPResponse` you
  configured, so serving one `bytes` object for every download keeps
  one copy no matter how many responses are recorded.
- The server keeps the most recent `recording_buffer_size` requests,
  responses and exchanges (256 by default).
- For each client address it keeps the most recent
  `max_connection_bytes` bytes sent and received (1 MiB by default).

So with the defaults, a stream of 1 GiB `Content-Length` uploads
keeps up to 256 GiB of request bodies.

## Bounding memory

```python
server = AsyncHTTPTestServer(
    recording_buffer_size=1,
    max_connection_bytes=0,
    max_recorded_body_bytes=1024,
)
```

- `recording_buffer_size=1` keeps only the latest request, response
  and exchange.
- `max_connection_bytes=0` turns off the per-connection byte log.
- `max_recorded_body_bytes=1024` reads every request body to its end,
  so framing and keep-alive work as usual, but keeps only its first
  1024 bytes.

With all three, memory stays flat however large the bodies are and
however many requests arrive.

`docs/examples/large_uploads.py` in the repository shows this with
64 MiB uploads.

## What a limited recording contains

When `max_recorded_body_bytes` cuts a body short:

- `request.body` is the first N bytes of the payload. Handlers and
  middleware see the same prefix.
- `request.full_body_size` is the number of payload bytes the client
  sent. It is `None` when the body was kept whole.
- `request.wire_raw_bytes` stops at the same payload byte, so later
  chunk framing and trailers, such as a trailing checksum, are not
  recorded.

The option cannot be combined with forwarding (`forward_proxy`,
`upstream_client` or `raw_forwarder`), which needs the whole request.

## Running a benchmark endpoint

The server runs on one asyncio event loop. Run it in its own process
so the client under test does not compete with it for the same core
and interpreter lock:

```python
import asyncio

from localstub import AsyncHTTPTestServer, HTTPResponse

OBJECT = bytes(8 * 1024 * 1024)


async def main() -> None:
    async with AsyncHTTPTestServer(
        port=8080,
        recording_buffer_size=1,
        max_connection_bytes=0,
        max_recorded_body_bytes=1024,
        default_response=HTTPResponse.raw(OBJECT),
    ) as server:
        print(f"listening on {server.url}")
        await asyncio.Event().wait()


asyncio.run(main())
```

The `lstub` CLI accepts `--max-recorded-body-bytes` and
`--write-chunk-size` too, but it prints every exchange to the console,
which costs time on each request.

## Large downloads over TLS

By default each response body goes out in one write. Over TLS, through
`AsyncTLSInterceptProxy`, that encrypts the whole body into buffers as
large as the body, which costs CPU in page faults and cache misses.
`write_chunk_size` sends each body as writes of at most that many bytes
and waits for each to drain, so the buffers stay small:

```python
server = AsyncHTTPTestServer(
    recording_buffer_size=1,
    max_connection_bytes=0,
    write_chunk_size=256 * 1024,
)
```

Whether it helps depends on the client. On a 2-core machine, with the
AWS CRT downloading 1 GiB in 8 MiB ranged GETs over 10 connections
through the TLS proxy, 256 KiB writes cut server CPU by about 37% and
raised throughput from about 9.5 to 11.5 Gbps. Sizes from 64 KiB to
512 KiB performed about the same. With 10 connections read by Python's
`ssl` module, server CPU fell by a similar amount but throughput fell
from about 5 to 3.4 Gbps, because less data is queued for each reader.
Measure with your own client before turning it on. Over plain HTTP it
does not help.
