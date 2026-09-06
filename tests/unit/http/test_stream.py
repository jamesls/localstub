from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio

from localstub.http.stream import read, take_unread_data, unread_data


@pytest_asyncio.fixture
async def reader() -> asyncio.StreamReader:
    return asyncio.StreamReader()


def test_take_unread_data_without_unread_returns_empty(
    reader: asyncio.StreamReader,
) -> None:
    assert take_unread_data(reader) == b""


def test_unread_data_with_empty_data_holds_nothing(
    reader: asyncio.StreamReader,
) -> None:
    unread_data(reader, b"")

    assert take_unread_data(reader) == b""


def test_take_unread_data_returns_all_bytes_by_default(
    reader: asyncio.StreamReader,
) -> None:
    unread_data(reader, b"hello")

    assert take_unread_data(reader) == b"hello"
    assert take_unread_data(reader) == b""


def test_take_unread_data_with_limit_keeps_remainder(
    reader: asyncio.StreamReader,
) -> None:
    unread_data(reader, b"hello world")

    assert take_unread_data(reader, 5) == b"hello"
    assert take_unread_data(reader) == b" world"


def test_unread_data_prepends_ahead_of_earlier_unread(
    reader: asyncio.StreamReader,
) -> None:
    unread_data(reader, b"second")
    unread_data(reader, b"first")

    assert take_unread_data(reader) == b"firstsecond"


@pytest.mark.asyncio
async def test_read_returns_unread_bytes_before_stream_data() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"stream data")
    reader.feed_eof()
    unread_data(reader, b"held")

    assert await read(reader, 1024) == b"held"
    assert await read(reader, 1024) == b"stream data"


@pytest.mark.asyncio
async def test_read_with_negative_size_returns_all_data_in_order() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(b"stream data")
    reader.feed_eof()
    unread_data(reader, b"held ")

    assert await read(reader, -1) == b"held stream data"


@pytest.mark.asyncio
async def test_read_returns_unread_bytes_after_eof() -> None:
    reader = asyncio.StreamReader()
    reader.feed_eof()
    unread_data(reader, b"abcdef")

    assert await read(reader, 4) == b"abcd"
    assert await read(reader, 4) == b"ef"
    assert await read(reader, 4) == b""
