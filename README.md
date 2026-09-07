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

[Documentation](https://jamesls.github.io/localstub/)

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

### Mutation testing

[mutmut](https://mutmut.readthedocs.io/en/latest/) is included in the dev
dependencies. It changes small pieces of code and checks whether the tests
fail: a "killed" mutant was caught; a "survived" mutant needs inspection.
It requires fork support (Linux/macOS, or WSL on Windows).

```sh
# Run the full source scan with four workers (can take a while).
uv run poe mutate

# Also exercise property and other decorated bodies in an isolated copy.
uv run poe mutate-decorated

# Start with a smaller module instead; quote patterns to avoid shell expansion.
uv run poe mutate 'localstub.http.responsespec.*'

# Inspect results and individual diffs, or open the interactive browser.
uv run mutmut results
uv run mutmut show localstub.http.responsespec.x__with_defaults__mutmut_2
uv run mutmut browse
```

Mutation runs use `tests/`, with pytest coverage disabled to avoid collecting
coverage on every mutant. Normal `poe test` coverage is unchanged. Generated
code and cached results live in the git-ignored `mutants/` directory. After
changing tests, explicitly rerun the relevant mutant name or module pattern
with `poe mutate` to refresh its results. Mutation testing is deliberately
separate from `prcheck`; survivors are not automatically bugs.

The uv lock pins an upstream fix for mutmut's decorated-class exclusion;
`HTTPResponse`, `Headers`, `Router`, and the other dataclasses now generate
mutants. The supplemental runner exposes decorator-hidden bodies only in
the git-ignored `mutants-decorated/` copy, preserving the original library.
It starts fresh each time so renamed helpers cannot reuse stale test tracking.
Inspect its separate results with:

```sh
uv run python scripts/mutate_decorated.py results
uv run python scripts/mutate_decorated.py browse
```

See [the mutation-testing report](docs/mutation-testing.md) for the setup,
coverage limitations, and findings. A completed scan is not a guarantee that
every source construct can be mutated, or that every survivor is a bug.
