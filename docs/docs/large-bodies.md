# Large bodies and benchmarking

localstub records every request it receives so that a test can assert on it
afterwards. For large uploads, such as multi-gigabyte objects, or for a
throughput benchmark, those recordings account for most of the server's memory
use. This page covers the settings that bound that memory, and an option that
cuts server CPU use for large downloads over TLS.

## Recorded data

- Each recorded request holds its body. For a `Content-Length` body,
  `wire_raw_bytes` reuses the body's bytes and builds its value only
  when you read it, so the request holds one copy of the body. For a
  chunked body, the wire bytes include the chunk framing, so the
  request holds them in addition to the body.
- Each recorded response refers to the body of the `HTTPResponse` you
  configured instead of copying it. If every download serves the same
  `bytes` object, the server holds one copy of it however many
  responses it records.
- The server keeps the most recent `recording_buffer_size` requests,
  responses and exchanges (256 by default).
- For each client address, the server keeps a byte log of the most
  recent `max_connection_bytes` bytes sent and received (1 MiB by
  default).

With the defaults, a stream of 1 GiB `Content-Length` uploads keeps up
to 256 GiB of request bodies.

## Bound memory use

```python
server = AsyncHTTPTestServer(
    recording_buffer_size=1,
    max_connection_bytes=0,
    max_recorded_body_bytes=1024,
)
```

- `recording_buffer_size=1` keeps only the latest request, response
  and exchange.
- `max_connection_bytes=0` turns off the byte log.
- `max_recorded_body_bytes=1024` keeps only the first 1024 bytes of
  each request body. The server still reads every body to its end, so
  framing and keep-alive work as usual.

With all three settings, memory use stays constant regardless of body
size or request count. The repository's
`docs/examples/large_uploads.py` shows this configuration with 64 MiB
uploads.

## Truncated request bodies

If a request body is longer than `max_recorded_body_bytes`, the server
truncates the recorded body:

- `request.body` holds the first `max_recorded_body_bytes` bytes of
  the payload. Handlers and middleware see the same prefix.
- `request.full_body_size` is the number of payload bytes the client
  sent. If the server kept the whole body, it is `None`.
- `request.wire_raw_bytes` ends at the same payload byte. The server
  does not record the chunk framing or trailers that follow, such as
  a trailing checksum.

Forwarding needs the whole request, so `max_recorded_body_bytes`
cannot be combined with `forward_proxy`, `upstream_client` or
`raw_forwarder`. The constructor raises `ValueError` if you pass both.

## Run a benchmark endpoint

The server runs on one asyncio event loop. Run it in its own process
so the client under test does not compete with it for the same CPU
core and the interpreter lock:

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

The `lstub` CLI also accepts `--max-recorded-body-bytes` and
`--write-chunk-size`. The CLI prints every exchange to the console,
which adds time to each request, so use a script like the one above
for benchmarks.

## Large TLS downloads

By default, the server sends each response body in one write. Through
`AsyncTLSInterceptProxy`, that write encrypts the whole body into
buffers as large as the body, and filling those buffers costs CPU in
page faults and cache misses. If you set `write_chunk_size`, the
server splits each body into writes of at most that many bytes and
waits for each write to drain before it sends the next. The buffers
then stay small:

```python
server = AsyncHTTPTestServer(
    recording_buffer_size=1,
    max_connection_bytes=0,
    write_chunk_size=256 * 1024,
)
```

`write_chunk_size` helps only over TLS, and only with some clients. In
one test on a 2-core machine, the AWS CRT downloaded 1 GiB through the
TLS proxy in 8 MiB ranged GETs over 10 connections. With 256 KiB
writes, server CPU use fell by about 37% and throughput rose from
about 9.5 to 11.5 Gbps. Write sizes from 64 KiB to 512 KiB gave about
the same results. A client that read 10 connections with Python's
`ssl` module saw server CPU use fall by a similar amount, but its
throughput fell from about 5 to 3.4 Gbps because less data is queued
for each reader.

Measure with your own client before you turn on `write_chunk_size`.
