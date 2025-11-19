# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

localstub is a Python library providing an asyncio-based HTTP test server for testing SDK clients. It records raw HTTP requests (including chunked/aws-chunked framing and trailers) and allows configurable responses.

## Development Environment

- **Python Version**: 3.12 (strictly, not 3.13+)
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
uv run pytest tests/unit/test_server.py

# Run specific test function
uv run pytest tests/integration/test_server.py::test_server_receives_request_and_returns_json_response

# Run tests with verbose output
uv run pytest -v tests/
```

### Code Quality
```sh
# Run all checks (linting, formatting, type checking)
poe check

# Auto-fix formatting and linting issues
poe auto-check

# Format code only
poe format-code

# Before submitting PR
poe prcheck  # runs check + test
```

## Code Architecture

### Core Components

**AsyncHTTPTestServer** (`src/localstub/server.py`)
- Main test server class using asyncio's `start_server`
- Records all incoming requests in `.requests` list and `.last_request`
- Supports both static responses and dynamic handler functions
- Handler can be sync or async: `Callable[[RequestRecorder], Awaitable[StubResponse] | StubResponse]`

**RequestRecorder** (`src/localstub/server.py`)
- Captures HTTP request details including method, path, headers, body
- `wire_raw_bytes` contains exact bytes received from the wire, including chunked framing
- `json_body` property for convenient JSON access
- `client` tuple contains (host, port) of the client

**StubResponse** (`src/localstub/server.py`)
- Dataclass for configuring HTTP responses
- Factory methods: `.json()`, `.text()`, `.raw()` for common response types
- Automatically sets appropriate Content-Type and Content-Length headers

### Key Features

1. **Raw Wire Bytes**: The server preserves exact bytes received, including Transfer-Encoding chunked framing and AWS chunked signatures
2. **Dynamic Handlers**: Can set `.handler` property to a function for request-specific responses
3. **Static Responses**: Use `.set_json_response()`, `.set_text_response()`, or `.set_raw_response()` for fixed responses
4. **Async Context Manager**: Use `async with AsyncHTTPTestServer() as server:` for automatic cleanup
5. **Request Queue**: `.next_request(timeout=...)` allows awaiting the next incoming request

### Project Structure

```
src/localstub/
  __init__.py      # Empty, package marker
  server.py        # All core functionality

tests/
  unit/            # Unit tests
  integration/     # Integration tests with httpx client
```

## Code Style

- **Line length**: 79 characters (enforced by ruff)
- **Quote style**: Preserve existing quotes (configured in ruff)
- **Type hints**: Required (pyright enforces this)
- **Imports**: `from __future__ import annotations` for modern type hints

## Coverage

- Coverage tracking configured via pytest-cov
- Target: 100% coverage on all new code
- Branch coverage enabled
- Excludes: `TYPE_CHECKING` blocks, `pragma: no cover`, `raise NotImplementedError`
