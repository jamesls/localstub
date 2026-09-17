from __future__ import annotations

import asyncio
from unittest.mock import create_autospec

import pytest

from localstub.forward import ClientWriter, ForwardError, RawForwarder
from localstub.http.request import AsyncRequestParser, RecordedHTTPRequest
from localstub.server import DropConnection
from localstub.tlsproxy import fault_step_transformer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("framing_header", "body_wire"),
    [
        (b"Content-Length: 10\r\n", b"abc"),
        (b"Transfer-Encoding: chunked\r\n", b"3\r\nabc\r\n"),
    ],
    ids=["content-length", "chunked"],
)
async def test_forward_with_100_continue_preserves_interrupted_upload_wire(
    framing_header: bytes,
    body_wire: bytes,
) -> None:
    header_wire = (
        b"PUT /upload HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        + framing_header
        + b"Expect: 100-continue\r\n\r\n"
    )
    client_reader = asyncio.StreamReader()
    client_reader.feed_data(header_wire)
    parser = AsyncRequestParser()
    headers, remaining = await parser.parse_headers(client_reader)
    assert headers.parsed is not None

    client_reader.feed_data(body_wire)
    client_reader.feed_eof()
    upstream_reader = asyncio.StreamReader()
    continue_wire = b"HTTP/1.1 100 Continue\r\n\r\n"
    upstream_reader.feed_data(continue_wire)
    upstream_reader.feed_eof()
    client_writer = create_autospec(asyncio.StreamWriter, instance=True)
    upstream_writer = create_autospec(asyncio.StreamWriter, instance=True)

    forwarded = await RawForwarder().forward_with_100_continue(
        parser=parser,
        parsed_headers=headers.parsed,
        header_wire_bytes=headers.wire_bytes,
        remaining_buffer=remaining,
        client_reader=client_reader,
        client_writer=client_writer,
        upstream_reader=upstream_reader,
        upstream_writer=upstream_writer,
        request_method="PUT",
    )

    assert forwarded.error is ForwardError.REQUEST_PARSE_FAILED
    assert not forwarded.request_body_consumed
    assert forwarded.response is None
    recorded = RecordedHTTPRequest.from_parsed(
        forwarded.parsed_request,
        forwarded.request_wire_bytes,
    )
    assert recorded.body == b"abc"
    assert recorded.wire_raw_bytes == header_wire + body_wire
    client_writer.write.assert_called_once_with(continue_wire)
    upstream_writer.write.assert_called_once_with(header_wire)


@pytest.mark.asyncio
@pytest.mark.parametrize("reset", [False, True])
async def test_forward_fault_adapter_uses_requested_closure(
    reset: bool,
) -> None:
    upstream_reader = asyncio.StreamReader()
    upstream_reader.feed_data(
        b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nabc"
    )
    upstream_reader.feed_eof()
    writer = create_autospec(ClientWriter, instance=True)
    forwarder = RawForwarder(
        response_transformer=fault_step_transformer(
            DropConnection(after_bytes=5, reset=reset)
        )
    )

    result = await forwarder.read_and_relay_responses(
        upstream_reader=upstream_reader,
        client_writer=writer,
        request_method="GET",
    )

    assert result is not None
    writer.write.assert_called_once_with(b"HTTP/")
    writer.drain.assert_awaited_once()
    writer.wait_closed.assert_awaited_once()
    if reset:
        writer.reset.assert_called_once()
        writer.close.assert_not_called()
    else:
        writer.close.assert_called_once()
        writer.reset.assert_not_called()
