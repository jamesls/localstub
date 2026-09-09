from __future__ import annotations

from pytest_codspeed import BenchmarkFixture

from localstub.recording import DEFAULT_MAX_CONNECTION_BYTES, BoundedByteBuffer

FILL = bytes(DEFAULT_MAX_CONNECTION_BYTES)
WRITE = b"x" * 64
WRITE_COUNT = 4096


def test_bounded_byte_buffer_churn_at_capacity(
    benchmark: BenchmarkFixture,
) -> None:
    # Once the buffer is full every append evicts from the front, which
    # is the path a per-connection recording buffer lives on for the
    # lifetime of a long connection.
    def churn() -> bytes:
        buffer = BoundedByteBuffer(DEFAULT_MAX_CONNECTION_BYTES)
        buffer.extend(FILL)
        for _ in range(WRITE_COUNT):
            buffer.extend(WRITE)
        return bytes(buffer)

    retained = benchmark(churn)

    assert len(retained) == DEFAULT_MAX_CONNECTION_BYTES
    assert retained.endswith(WRITE * WRITE_COUNT)
