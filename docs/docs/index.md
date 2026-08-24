# localstub

localstub is a Python library and CLI for testing HTTP clients against
a real server. It runs an asyncio HTTP server in your test process,
returns responses you configure (including errors, slow responses, and
disconnects), and records each request exactly as it arrived on the
wire so your tests can inspect it.

localstub also includes a TLS intercept proxy so you can test a client
without changing its code. Set the `HTTPS_PROXY` and CA bundle
environment variables that most HTTP clients already honor, and the
client's traffic goes to the stub instead of the real server. This
also lets you see the exact bytes an existing client sends.

!!! warning "Pre-1.0 software"

    localstub is under active development. There may be breaking API
    changes until the 1.0.0 GA release.

## Example

An unmodified `httpx` client makes a request to `https://example.com/`.
The proxy routes the request to a local test server that returns a
canned response and records what the client sent:

```python
import asyncio
import ssl

import httpx

from localstub import AsyncHTTPTestServer, AsyncTLSInterceptProxy


async def main() -> None:
    async with (
        AsyncHTTPTestServer() as server,
        AsyncTLSInterceptProxy(server=server) as proxy,
    ):
        # Configure the test server with whatever response you want.
        server.set_json_response({"ok": True})

        # This example uses httpx, but in your tests this would
        # be your own client library.
        async with httpx.AsyncClient(
            proxy=proxy.endpoint_url,
            verify=ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            ),
        ) as client:
            # A normal request, except the proxy routes it to the
            # test server, which returns the response set above.
            response = await client.get("https://example.com/")
            assert response.json() == {"ok": True}

        # The server records every request it receives in
        # `server.requests`.
        request = server.requests[0]
        assert request.headers["host"] == "example.com"


asyncio.run(main())
```

## Comparison with client-side mocking

Mocking at the client layer (patching your HTTP library, or swapping
in a fake transport) verifies that your code *called* the client. It
doesn't verify what goes on the wire. With localstub your client does
everything for real: it resolves, connects, serializes headers, frames
the body, and parses the response. That's where bugs like incorrect
chunked framing, missing headers, or broken retry logic live.

## Installation

```sh
uv add localstub
```

or with pip:

```sh
pip install localstub
```

## Next steps

- [Getting started](getting-started.md): write your first test
  against a plain-HTTP stub server.
