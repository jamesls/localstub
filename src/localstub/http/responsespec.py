from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HTTPResponse:
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes | str = b""

    @classmethod
    def json(
        cls,
        obj: Any,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> HTTPResponse:
        text = json.dumps(obj)
        body = text.encode("utf-8")
        base_headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(body)),
        }
        if headers:
            base_headers.update(headers)
        return cls(status=status, headers=base_headers, body=body)

    @classmethod
    def text(
        cls,
        text: str,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> HTTPResponse:
        body = text.encode("utf-8")
        base_headers = {
            "Content-Type": "text/plain; charset=utf-8",
            "Content-Length": str(len(body)),
        }
        if headers:
            base_headers.update(headers)
        return cls(status=status, headers=base_headers, body=body)

    @classmethod
    def raw(
        cls,
        data: bytes,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> HTTPResponse:
        base_headers = {"Content-Length": str(len(data))}
        if headers:
            base_headers.update(headers)
        return cls(status=status, headers=base_headers, body=data)
