from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from localstub.server import HTTPResponse


@dataclass
class ResponseConfig:
    """Parsed response configuration from a JSON config file."""

    single_response: HTTPResponse | None
    response_sequence: list[HTTPResponse] | None


def load_config(config_path: Path) -> ResponseConfig:
    """Load and parse a JSON config file.

    Args:
        config_path: Path to the JSON config file

    Returns:
        ResponseConfig with either single_response or response_sequence set

    Raises:
        ValueError: If config file has invalid structure
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file contains invalid JSON
    """
    with open(config_path) as f:
        data = json.load(f)

    return _parse_config(data)


def _parse_config(data: dict[str, Any]) -> ResponseConfig:
    """Parse config dict into ResponseConfig."""
    if "responses" in data:
        responses = [_parse_response_spec(r) for r in data["responses"]]
        return ResponseConfig(
            single_response=None, response_sequence=responses
        )
    elif "response" in data:
        response = _parse_response_spec(data["response"])
        return ResponseConfig(single_response=response, response_sequence=None)
    else:
        raise ValueError(
            "Config must have either 'response' or 'responses' key"
        )


def _parse_response_spec(spec: dict[str, Any]) -> HTTPResponse:
    """Parse a single response specification into HTTPResponse."""
    resp_type = spec.get("type", "json")
    status = spec.get("status", 200)
    headers = spec.get("headers")
    body = spec.get("body")
    encoding = spec.get("encoding", "utf-8")

    if resp_type == "json":
        return HTTPResponse.json(body, status=status, headers=headers)
    elif resp_type == "text":
        text = body if isinstance(body, str) else str(body) if body else ""
        return HTTPResponse.text(text, status=status, headers=headers)
    elif resp_type == "raw":
        raw_bytes: bytes
        if body is None:
            raw_bytes = b""
        elif encoding == "base64":
            if not isinstance(body, str):
                raise ValueError("base64 body must be a string")
            raw_bytes = base64.b64decode(body)
        elif isinstance(body, str):
            raw_bytes = body.encode("utf-8")
        elif isinstance(body, bytes):
            raw_bytes = body
        else:
            msg = f"Invalid body type for raw response: {type(body)}"
            raise ValueError(msg)
        return HTTPResponse.raw(raw_bytes, status=status, headers=headers)
    else:
        raise ValueError(f"Unknown response type: {resp_type}")
