# Getting started

This page shows how to install localstub and write your first test:
stub a JSON response, make a request with a real HTTP client, and
assert on the request the server recorded. You need Python 3.12 or
newer.

## Installation

```sh
uv add localstub
```

or with pip:

```sh
pip install localstub
```

The example below uses [httpx](https://www.python-httpx.org/) as the
client, but any HTTP client works. In your own tests, point the
client you are testing at the stub.

## Your first test

Save this as `example.py`:

```python
import asyncio

import httpx

from localstub import AsyncHTTPTestServer


async def main() -> None:
    async with AsyncHTTPTestServer() as server:
        # Configure the response the server returns.
        server.set_json_response({"status": "ok"})

        # Point a real HTTP client at the server.
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{server.url}status")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

        # The server recorded the request the client sent.
        request = server.last_request
        assert request is not None
        assert request.method == "GET"
        assert request.target == "/status"
        assert request.headers["host"] is not None


asyncio.run(main())
```

## Run the example

The example uses `httpx`, which is not installed with localstub. Use
uv to provide it for this command:

```sh
uv run --with httpx python example.py
```

## How it works

- `AsyncHTTPTestServer()` started a real asyncio HTTP server on a
  random local port. Its base URL is `server.url`, and the
  `async with` block shuts the server down on exit.
- `server.set_json_response({"status": "ok"})` configured a static
  response: every request receives HTTP 200 with a JSON body and
  `Content-Type: application/json`.
- The `httpx` client made a real HTTP request over a real socket.
  Nothing was mocked or patched.
- The server recorded the request. `server.last_request` holds the
  most recent one and `server.requests` holds all of them, so you
  can assert on the method, target, headers, and body your client
  sent.

## Next steps

This example used plain HTTP against `server.url`. From here you can:

- Configure richer responses: error statuses, response sequences for
  retry testing, and dynamic handlers.
- Test an unmodified client over HTTPS with the TLS intercept proxy
  (see the example on the [landing page](index.md)).
- Inspect the raw wire bytes of each request, including chunked
  framing and trailers, via `request.wire_raw_bytes`.
