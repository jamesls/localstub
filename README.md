# localstub

localstub is a Python library and CLI for testing HTTP clients against
the behavior of a real server.  It runs an asyncio HTTP server in
your test process, returns whatever responses you configure, including
error responses, slow responses, disconnects, and records each request
exactly as it arrived on the wire for your tests to inspect.

A TLS proxy is provided so you can test your clients without having to
make any config changes, most HTTP clients support `HTTPS_PROXY` and
some environment variables for specifying a `CA_BUNDLE`.  This also
lets you inspect existing HTTP clients to see the exact bytes they
are sending to servers.

> [!WARNING]
> localstub is under active development.  There may be breaking API changes
> until the 1.0.0 GA release.


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

        # Using a generic HTTP client here, but this would be YOUR
        # client library that's making calls to your remote service.
        async with httpx.AsyncClient(
            proxy=proxy.endpoint_url,
            verify=ssl.create_default_context(
                cafile=str(proxy.ca.ca_pem_path())
            ),
        ) as client:
            # Your client makes a request just like it normally
            # would, but it gets routed to the test server we've
            # setup which will return an `{"ok": true}` JSON response.
            response = await client.get("https://example.com/")
            assert response.json() == {"ok": True}

        # The server records every request it receives in
        # `server.requests`, so you can inspect the exact
        # request the client sent.
        request = server.requests[0]
        assert request.headers["host"] == "example.com"


asyncio.run(main())
```

## Installation

```sh
uv add localstub
```

## Development

This project requires Python 3.12 and uses
[uv](https://github.com/astral-sh/uv) to manage dependencies.

You can create a virtual environment with all the necessary dependencies
by running:

```sh
uv sync --all-extras --dev
```

This will install all necessary dependencies and install this project
in editable mode.

You can activate the venv with:

```sh
. .venv/bin/activate
```

### Testing

[Poe the Poet](https://github.com/nat-n/poethepoet) is the task runner
used for this project, it's automatically installed as part of the
dev dependencies.  To see a list of available tasks, run the
`poe` command with no args.

To run the tests for this project run:


```sh
poe test
```

Before submitting a PR, ensure the `prcheck` task runs successfully:

```sh
poe prcheck
```
