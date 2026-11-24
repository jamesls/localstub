# Project Overview

localstub is a Python library providing an asyncio-based HTTP test server for testing SDK clients. It records raw HTTP requests (including chunked framing and trailers) and allows configurable responses.

## Development Environment

- **Python Version**: 3.12 through 3.14 (CI tests all three)
- **Package Manager**: [uv](https://github.com/astral-sh/uv)
- **Task Runner**: [Poe the Poet](https://github.com/nat-n/poethepoet)

### Setup

```sh
# Create virtual environment with all dependencies
uv sync --all-extras --dev

# Activate virtual environment
. .venv/bin/activate
```

## Common Commands

### Testing
```sh
# Run all tests with coverage
poe test

# Run specific test file
uv run pytest tests/unit/server/test_core.py

# Run specific test function
uv run pytest tests/integration/test_server.py::test_server_receives_request_and_returns_json_response

# Run tests with verbose output
uv run pytest -v tests/
```

### Code Quality

Run `poe auto-check` after every edit, including test changes and resolved
merge or rebase conflicts, and before any git operation that records the
result (commit, `rebase --continue`, `merge --continue`). It auto-fixes
formatting and lint, then runs pyrefly, ast-grep, and import-linter.

### Test Discipline

For any change beyond docs or comments, run the full suite with `poe test`
before handoff. The sequence is: change → `poe auto-check` → `poe test`.

## Code Architecture

### Core Components

**AsyncHTTPTestServer** (`src/localstub/server/core.py`)
- Main test server class using asyncio's `start_server`
- Records all incoming requests in `.requests` list and `.last_request`
- Supports both static responses and dynamic handler functions
- Handler (`ResponderHandler`, `src/localstub/router.py`) can be sync or
  async: `Callable[[ResponderContext], ResponseSpec | Awaitable[ResponseSpec]]`
- Keep-alive policy via `set_keep_alive(...)`; connection faults via
  `CloseConnection`, `CloseDuringRequest`, and `DropConnection`

**HTTPConnection** (`src/localstub/server/connection.py`)
- Runs the per-connection request loop and owns a `ConnectionState`
- Publishes exactly one `ConnectionClosed` event per connection
  (`.closed_connections`, `.last_closed_connection`, `.next_closed_connection()`)

**HTTPRequest / RecordedHTTPRequest** (`src/localstub/http/request.py`)
- `HTTPRequest` is the RFC 9110 request: method, target, headers, body;
  `json_body` property for convenient JSON access
- `RecordedHTTPRequest` is what `.requests` / `.last_request` return: it
  wraps the request with `wire_raw_bytes` (exact bytes off the wire,
  including chunked framing) and `client` (host, port)

**HTTPResponse** (`src/localstub/http/responsespec.py`)
- Dataclass for configuring HTTP responses
- Factory methods: `.json()`, `.text()`, `.raw()` for common response types
- Automatically sets appropriate Content-Type and Content-Length headers

**WebSockets** (`src/localstub/websocket/`, `src/localstub/server/websocket.py`)
- `AcceptWebSocket` (`middleware/core.py`) is a `ResponseSpec`; the
  sender terminal writes the 101 and `_serve()` runs a
  `WebSocketSession` once the handshake exchange is recorded
- The package does no server imports: `frames` (codec), `deflate`
  (RFC 7692), `handshake`, `records`, `assembly` (violation checks,
  shared by sessions and the forward relay), `session`, `relay`
- Sessions decide closes through a callback, so `ConnectionState` stays
  the only place a close is decided; phase is `websocket`

### Key Features

1. **Raw Wire Bytes**: The server preserves exact bytes received, including Transfer-Encoding chunked framing and AWS chunked signatures
2. **Dynamic Handlers**: Can set `.handler` property to a function for request-specific responses
3. **Static Responses**: Use `.set_json_response()`, `.set_text_response()`, or `.set_raw_response()` for fixed responses
4. **Async Context Manager**: Use `async with AsyncHTTPTestServer() as server:` for automatic cleanup
5. **Request Queue**: `.next_request(timeout=...)` allows awaiting the next incoming request

## Code Style

- **Line length**: 79 characters (enforced by ruff)
- **Quote style**: Preserve existing quotes (configured in ruff)
- **Type hints**: Required (pyrefly enforces this in strict mode)
- **Imports**: `from __future__ import annotations` for modern type hints

Don't use `# type: ignore` or `# noqa`; fix the underlying issue instead.
`rules/no-suppression-comments.yml` rejects them in `poe auto-check`.

## Coverage

- Coverage tracking configured via pytest-cov
- Target: 100% coverage on all new code
- Branch coverage enabled
- Excludes: `TYPE_CHECKING` blocks, `pragma: no cover`, `raise NotImplementedError`
