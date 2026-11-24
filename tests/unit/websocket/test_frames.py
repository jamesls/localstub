from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.websocket.frames import (
    FrameDecoder,
    Opcode,
    WebSocketFrame,
    apply_mask,
    close_payload,
    parse_header,
)

MASK = bytes([0x37, 0xFA, 0x21, 0x3D])
# RFC 6455 section 5.7.
UNMASKED_HELLO = bytes([0x81, 0x05, 0x48, 0x65, 0x6C, 0x6C, 0x6F])
MASKED_HELLO = bytes([
    0x81,
    0x85,
    0x37,
    0xFA,
    0x21,
    0x3D,
    0x7F,
    0x9F,
    0x4D,
    0x51,
    0x58,
])
FRAGMENTED_HELLO = (
    bytes([0x01, 0x03, 0x48, 0x65, 0x6C]),
    bytes([0x80, 0x02, 0x6C, 0x6F]),
)
UNMASKED_PING = bytes([0x89, 0x05, 0x48, 0x65, 0x6C, 0x6C, 0x6F])
MASKED_PONG = bytes([
    0x8A,
    0x85,
    0x37,
    0xFA,
    0x21,
    0x3D,
    0x7F,
    0x9F,
    0x4D,
    0x51,
    0x58,
])


def _repeat(seed: bytes, length: int) -> bytes:
    # Large payloads span the 16- and 64-bit length forms without asking
    # Hypothesis to draw 64 KiB, which its health check rejects.
    return (seed * length)[:length]


def frames() -> st.SearchStrategy[WebSocketFrame]:
    return st.builds(
        WebSocketFrame,
        opcode=st.integers(0, 15),
        payload=st.one_of(
            st.binary(max_size=200),
            st.builds(
                _repeat,
                st.binary(min_size=1, max_size=4),
                st.integers(65530, 65540),
            ),
        ),
        fin=st.booleans(),
        rsv1=st.booleans(),
        rsv2=st.booleans(),
        rsv3=st.booleans(),
        mask_key=st.one_of(st.none(), st.binary(min_size=4, max_size=4)),
    )


@pytest.mark.parametrize(
    ("frame", "wire"),
    [
        (WebSocketFrame.text("Hello"), UNMASKED_HELLO),
        (WebSocketFrame(Opcode.TEXT, b"Hello", mask_key=MASK), MASKED_HELLO),
        (WebSocketFrame.text("Hel", fin=False), FRAGMENTED_HELLO[0]),
        (WebSocketFrame.continuation(b"lo"), FRAGMENTED_HELLO[1]),
        (WebSocketFrame.ping(b"Hello"), UNMASKED_PING),
        (WebSocketFrame(Opcode.PONG, b"Hello", mask_key=MASK), MASKED_PONG),
        (
            WebSocketFrame.binary(bytes(256)),
            b"\x82\x7e\x01\x00" + bytes(256),
        ),
        (
            WebSocketFrame.binary(bytes(65536)),
            b"\x82\x7f" + (65536).to_bytes(8, "big") + bytes(65536),
        ),
    ],
)
def test_encode_matches_rfc_6455_examples(
    frame: WebSocketFrame, wire: bytes
) -> None:
    assert frame.encode() == wire


def test_encode_uses_two_byte_length_up_to_65535() -> None:
    wire = WebSocketFrame.binary(bytes(65535)).encode()

    assert wire[:4] == b"\x82\x7e\xff\xff"


def test_encode_sets_reserved_bits_and_reserved_opcode() -> None:
    frame = WebSocketFrame(
        0x3, b"", fin=False, rsv1=True, rsv2=True, rsv3=True
    )

    assert frame.encode() == b"\x73\x00"


def test_decode_masked_rfc_example_keeps_wire_and_mask() -> None:
    [decoded] = FrameDecoder().feed(MASKED_HELLO)

    assert decoded.frame == WebSocketFrame(
        Opcode.TEXT, b"Hello", mask_key=MASK
    )
    assert decoded.wire_raw_bytes == MASKED_HELLO
    assert decoded.header.minimal_length


def test_decode_names_defined_opcodes_and_keeps_reserved_ones_as_int() -> None:
    decoder = FrameDecoder()
    [ping, reserved] = decoder.feed(UNMASKED_PING + b"\x8b\x00")

    assert ping.frame.opcode is Opcode.PING
    assert reserved.frame.opcode == 0xB
    assert not isinstance(reserved.frame.opcode, Opcode)


def test_decode_holds_partial_frame_until_complete() -> None:
    decoder = FrameDecoder()

    assert decoder.feed(MASKED_HELLO[:7]) == []
    assert decoder.buffered == 7
    header = decoder.partial_header
    assert header is not None
    assert header.length == 5
    [decoded] = decoder.feed(MASKED_HELLO[7:])
    assert decoded.frame.payload == b"Hello"
    assert decoder.buffered == 0
    assert decoder.partial_header is None


@pytest.mark.parametrize(
    "wire",
    [
        b"\x82\x7e\x00\x05" + bytes(5),
        b"\x82\x7f" + (300).to_bytes(8, "big") + bytes(300),
    ],
)
def test_decode_flags_non_minimal_length(wire: bytes) -> None:
    [decoded] = FrameDecoder().feed(wire)

    assert not decoded.header.minimal_length
    assert decoded.wire_raw_bytes == wire
    assert decoded.frame.encode() != wire


def test_parse_header_flags_64_bit_length_with_top_bit_set() -> None:
    header = parse_header(b"\x82\x7f\x80" + bytes(7))

    assert header is not None
    assert not header.minimal_length


@pytest.mark.parametrize(
    "prefix",
    [b"\x81", b"\x81\x7e\x00", b"\x81\x7f" + bytes(7), b"\x81\x85\x00\x00"],
)
def test_parse_header_needs_whole_header(prefix: bytes) -> None:
    assert parse_header(prefix) is None


@given(frames())
def test_decode_of_encode_returns_same_frame(frame: WebSocketFrame) -> None:
    wire = frame.encode()

    [decoded] = FrameDecoder().feed(wire)

    assert decoded.frame == frame
    assert decoded.wire_raw_bytes == wire
    assert decoded.header.minimal_length


@given(
    st.lists(frames(), min_size=1, max_size=5),
    st.lists(st.integers(1, 70000), max_size=10),
)
def test_any_split_of_stream_yields_same_frames(
    sent: list[WebSocketFrame], cuts: list[int]
) -> None:
    wire = b"".join(frame.encode() for frame in sent)
    offsets = sorted({cut % (len(wire) + 1) for cut in cuts})
    decoder = FrameDecoder()

    decoded = []
    for start, end in zip([0, *offsets], [*offsets, len(wire)], strict=True):
        decoded.extend(decoder.feed(wire[start:end]))

    assert [item.frame for item in decoded] == sent
    assert b"".join(item.wire_raw_bytes for item in decoded) == wire
    assert decoder.buffered == 0


@given(st.binary(max_size=300), st.binary(min_size=4, max_size=4))
def test_masking_twice_returns_original(payload: bytes, key: bytes) -> None:
    masked = apply_mask(payload, key)

    assert len(masked) == len(payload)
    assert apply_mask(masked, key) == payload


def test_apply_mask_matches_bytewise_xor() -> None:
    payload = bytes(range(11))

    expected = bytes(b ^ MASK[i % 4] for i, b in enumerate(payload))
    assert apply_mask(payload, MASK) == expected


def test_close_factory_encodes_code_and_reason() -> None:
    frame = WebSocketFrame.close(1001, "bye")

    assert frame.opcode == Opcode.CLOSE
    assert frame.payload == b"\x03\xe9bye"
    assert WebSocketFrame.close(None).payload == b""


def test_close_payload_rejects_reason_without_code() -> None:
    with pytest.raises(ValueError, match="without a code"):
        close_payload(None, "why")


@pytest.mark.parametrize("code", [-1, 0x10000])
def test_close_payload_rejects_code_outside_16_bits(code: int) -> None:
    with pytest.raises(ValueError, match="16 bits"):
        close_payload(code)


def test_close_payload_sends_codes_that_must_not_appear_on_wire() -> None:
    assert close_payload(1005) == b"\x03\xed"


def test_frame_rejects_opcode_outside_4_bits() -> None:
    with pytest.raises(ValueError, match="4 bits"):
        WebSocketFrame(16)


def test_frame_rejects_mask_key_of_wrong_length() -> None:
    with pytest.raises(ValueError, match="4 bytes"):
        WebSocketFrame(Opcode.TEXT, mask_key=b"abc")


def test_factories_build_expected_frames() -> None:
    assert WebSocketFrame.binary(b"x", fin=False) == WebSocketFrame(
        Opcode.BINARY, b"x", fin=False
    )
    assert WebSocketFrame.pong(b"p") == WebSocketFrame(Opcode.PONG, b"p")


def test_partial_payload_unmasks_from_any_offset() -> None:
    payload = bytes(range(40))
    wire = WebSocketFrame(Opcode.BINARY, payload, mask_key=MASK).encode()
    decoder = FrameDecoder()
    decoder.feed(wire[:-5])

    assert decoder.partial_payload() == payload[:-5]
    assert decoder.partial_payload(7) == payload[7:-5]
    assert FrameDecoder().partial_payload() == b""


def test_partial_payload_of_unmasked_frame_is_raw() -> None:
    decoder = FrameDecoder()
    decoder.feed(WebSocketFrame.text("hello").encode()[:5])

    assert decoder.partial_payload(1) == b"el"
