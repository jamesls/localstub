from __future__ import annotations

import base64
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from typing import Literal, TextIO

from localstub.http.exchange import ConnectionClosed, RecordedExchange
from localstub.http.headers import Headers
from localstub.http.request import RecordedHTTPRequest
from localstub.http.response import RecordedHTTPResponse
from localstub.server import AsyncHTTPTestServer


def _to_base64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _safe_utf8_decode(data: bytes) -> str | None:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _isoformat_utc(ts: datetime) -> str:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC).isoformat()
    return ts.astimezone(UTC).isoformat()


def _client_json(
    client: tuple[str, int] | None,
) -> dict[str, object] | None:
    if client is None:
        return None
    host, port = client
    return {"host": host, "port": port}


def _headers_json(
    headers: Headers | Message | Mapping[str, str] | None,
) -> dict[str, str]:
    if headers is None:
        return {}
    return {str(k): str(v) for k, v in headers.items()}


def _request_body_json(request: RecordedHTTPRequest) -> str | None:
    if request.body is None:
        return None
    if not request.body:
        return ""
    return _safe_utf8_decode(request.body)


def _response_body_json(response: RecordedHTTPResponse) -> str | None:
    boundary = b"\r\n\r\n"
    wire = response.wire_raw_bytes
    idx = wire.find(boundary)
    if idx == -1:
        return _safe_utf8_decode(wire)
    body_bytes = wire[idx + len(boundary) :]
    return _safe_utf8_decode(body_bytes)


def _response_json(response: RecordedHTTPResponse) -> dict[str, object]:
    return {
        "status": response.status,
        "reason": response.reason,
        "headers": _headers_json(response.headers),
        "body": _response_body_json(response),
        "raw_wire_bytes": _to_base64(response.wire_raw_bytes),
    }


def _closed_json(closed: ConnectionClosed | None) -> dict[str, object] | None:
    if closed is None:
        return None
    return {
        "client": _client_json(closed.client),
        "reason": closed.reason,
        "phase": closed.phase,
        "reset": closed.reset,
        "requests_completed": closed.requests_completed,
        "bytes_read": closed.bytes_read,
        "bytes_consumed": closed.bytes_consumed,
        "bytes_written": closed.bytes_written,
        "timestamp": _isoformat_utc(closed.timestamp),
    }


def exchange_to_json_obj(exchange: RecordedExchange) -> dict[str, object]:
    request = exchange.request
    request_wire = request.wire_raw_bytes

    request_obj: dict[str, object] = {
        "method": request.method,
        "path": request.target,
        "client": _client_json(request.client),
        "headers": _headers_json(request.headers),
        "body": _request_body_json(request),
        "body_complete": request.body_complete,
        "raw_wire_bytes": _to_base64(request_wire),
    }
    if request.is_proxy_request and request.target_uri is not None:
        request_obj["target_host"] = request.target_uri.host
        request_obj["target_port"] = request.target_uri.port
        request_obj["effective_path"] = request.effective_path

    response_obj: dict[str, object] | None
    response_timestamp: str | None
    if exchange.response is None:
        response_obj = None
        response_timestamp = None
    else:
        response_obj = _response_json(exchange.response)
        response_timestamp = (
            _isoformat_utc(exchange.response_timestamp)
            if exchange.response_timestamp is not None
            else None
        )

    return {
        "timestamp": _isoformat_utc(exchange.request_timestamp),
        "request": request_obj,
        "response_timestamp": response_timestamp,
        "response": response_obj,
        "interim_responses": [
            _response_json(interim) for interim in exchange.interim_responses
        ],
        "closed": _closed_json(exchange.closed),
    }


class JSONLTrafficWriter:
    def __init__(self, fp: TextIO, *, flush_each: bool = True) -> None:
        self._fp = fp
        self._flush_each = flush_each

    def write_exchange(self, exchange: RecordedExchange) -> None:
        record = exchange_to_json_obj(exchange)
        self._fp.write(json.dumps(record) + "\n")
        if self._flush_each:
            self._fp.flush()


def dump_server_traffic_jsonl(
    server: AsyncHTTPTestServer,
    output: Path | TextIO,
    *,
    mode: Literal["w", "a"] = "w",
    start: int = 0,
    end: int | None = None,
) -> int:
    if start < 0:
        raise ValueError("start must be non-negative")
    if end is not None and end < 0:
        raise ValueError("end must be non-negative or None")

    exchanges = server.exchanges[start:end]

    if isinstance(output, Path):
        with output.open(mode, encoding="utf-8") as fp:
            writer = JSONLTrafficWriter(fp)
            for exchange in exchanges:
                writer.write_exchange(exchange)
        return len(exchanges)

    writer = JSONLTrafficWriter(output)
    for exchange in exchanges:
        writer.write_exchange(exchange)
    return len(exchanges)
