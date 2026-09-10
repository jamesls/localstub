from __future__ import annotations

from pytest_codspeed import BenchmarkFixture

from localstub.recording import DEFAULT_MAX_CONNECTION_BYTES, BoundedByteBuffer

FILL = bytes(DEFAULT_MAX_CONNECTION_BYTES)
SMALL_WRITE = b"x" * 64
SMALL_WRITE_COUNT = 4096
LARGE_WRITE = b"y" * (64 * 1024)
LARGE_WRITE_COUNT = 64
BODY_PIECE = b"z" * 8192
BODY_PIECE_COUNT = 64


def test_bounded_byte_buffer_churn_at_capacity(
    benchmark: BenchmarkFixture,
) -> None:
    # Once the buffer is full every append evicts from the front, which
    # is the path a per-connection recording buffer lives on for the
    # lifetime of a long connection.  Small writes are the common case:
    # a keep-alive connection carrying short requests and responses.
    def churn() -> bytes:
        buffer = BoundedByteBuffer(DEFAULT_MAX_CONNECTION_BYTES)
        buffer.extend(FILL)
        for _ in range(SMALL_WRITE_COUNT):
            buffer.extend(SMALL_WRITE)
        return bytes(buffer)

    retained = benchmark(churn)

    assert len(retained) == DEFAULT_MAX_CONNECTION_BYTES
    assert retained.endswith(SMALL_WRITE * SMALL_WRITE_COUNT)


def test_bounded_byte_buffer_large_writes_at_capacity(
    benchmark: BenchmarkFixture,
) -> None:
    # Large bodies flowing through a full buffer.  Each write displaces
    # whole chunks, so this guards against copying the retained bytes
    # on every eviction.
    def churn() -> bytes:
        buffer = BoundedByteBuffer(DEFAULT_MAX_CONNECTION_BYTES)
        buffer.extend(FILL)
        for _ in range(LARGE_WRITE_COUNT):
            buffer.extend(LARGE_WRITE)
        return bytes(buffer)

    retained = benchmark(churn)

    assert len(retained) == DEFAULT_MAX_CONNECTION_BYTES
    retained_writes = DEFAULT_MAX_CONNECTION_BYTES // len(LARGE_WRITE)
    assert retained == LARGE_WRITE * retained_writes


def test_bounded_byte_buffer_body_below_capacity(
    benchmark: BenchmarkFixture,
) -> None:
    # A large body recorded piecewise before the buffer fills, then
    # materialized once: the path for a single big upload or download.
    def record() -> bytes:
        buffer = BoundedByteBuffer(DEFAULT_MAX_CONNECTION_BYTES)
        for _ in range(BODY_PIECE_COUNT):
            buffer.extend(BODY_PIECE)
        return bytes(buffer)

    body = benchmark(record)

    assert body == BODY_PIECE * BODY_PIECE_COUNT
