"""RFC 6455 framing as values, with no I/O.

``WebSocketFrame`` is one frame and ``encode()`` its wire form.
``FrameDecoder`` is fed bytes as they arrive and returns each complete
frame together with the exact bytes that carried it, so a recording
keeps the mask key and the length encoding the peer chose.

Nothing here judges whether a frame is valid: a frame may carry a
reserved opcode, reserved bits, or an oversized control payload.  The
checks live in ``localstub.websocket.assembly``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple


class Opcode(IntEnum):
    """The six opcodes RFC 6455 defines."""

    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    CLOSE = 0x8
    PING = 0x9
    PONG = 0xA


CONTROL_OPCODES = frozenset({Opcode.CLOSE, Opcode.PING, Opcode.PONG})
DATA_OPCODES = frozenset({Opcode.CONTINUATION, Opcode.TEXT, Opcode.BINARY})
DEFINED_OPCODES = CONTROL_OPCODES | DATA_OPCODES

_LENGTH_16 = 126
_LENGTH_64 = 127
_TOP_BIT_64 = 1 << 63


def apply_mask(payload: bytes, mask_key: bytes) -> bytes:
    """XOR ``payload`` with the repeating four-byte ``mask_key``.

    The whole payload is XORed as one large integer, which runs in C,
    so a megabyte costs a few milliseconds rather than a Python loop
    per byte.  Masking twice with the same key returns the original.
    """
    length = len(payload)
    if not length:
        return b""
    key = (mask_key * (length // 4 + 1))[:length]
    value = int.from_bytes(payload, "big") ^ int.from_bytes(key, "big")
    return value.to_bytes(length, "big")


def close_payload(code: int | None, reason: str = "") -> bytes:
    """The payload of a Close frame: a 16-bit code and a UTF-8 reason.

    ``code=None`` gives an empty payload, which may not carry a reason.
    """
    if code is None:
        if reason:
            raise ValueError("a Close frame without a code has no reason")
        return b""
    if not 0 <= code <= 0xFFFF:
        raise ValueError(f"close code must fit in 16 bits, got {code}")
    return code.to_bytes(2, "big") + reason.encode("utf-8")


@dataclass(frozen=True)
class WebSocketFrame:
    """One RFC 6455 frame as a value.

    ``opcode`` is an ``int`` so a frame can carry a reserved opcode;
    ``Opcode`` names the six defined ones.  ``payload`` is the unmasked
    payload, and ``mask_key`` the four-byte key or ``None`` for an
    unmasked frame.
    """

    opcode: int
    payload: bytes = b""
    fin: bool = True
    rsv1: bool = False
    rsv2: bool = False
    rsv3: bool = False
    mask_key: bytes | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.opcode <= 0xF:
            raise ValueError(f"opcode must fit in 4 bits, got {self.opcode}")
        if self.mask_key is not None and len(self.mask_key) != 4:
            raise ValueError("mask_key must be exactly 4 bytes")

    @classmethod
    def text(cls, text: str, *, fin: bool = True) -> WebSocketFrame:
        return cls(Opcode.TEXT, text.encode("utf-8"), fin=fin)

    @classmethod
    def binary(cls, data: bytes, *, fin: bool = True) -> WebSocketFrame:
        return cls(Opcode.BINARY, data, fin=fin)

    @classmethod
    def continuation(cls, data: bytes, *, fin: bool = True) -> WebSocketFrame:
        return cls(Opcode.CONTINUATION, data, fin=fin)

    @classmethod
    def ping(cls, payload: bytes = b"") -> WebSocketFrame:
        return cls(Opcode.PING, payload)

    @classmethod
    def pong(cls, payload: bytes = b"") -> WebSocketFrame:
        return cls(Opcode.PONG, payload)

    @classmethod
    def close(
        cls, code: int | None = 1000, reason: str = ""
    ) -> WebSocketFrame:
        return cls(Opcode.CLOSE, close_payload(code, reason))

    def encode(self) -> bytes:
        """Return the wire bytes, using the shortest length form."""
        first = (
            (0x80 if self.fin else 0)
            | (0x40 if self.rsv1 else 0)
            | (0x20 if self.rsv2 else 0)
            | (0x10 if self.rsv3 else 0)
            | self.opcode
        )
        mask_bit = 0x80 if self.mask_key is not None else 0
        length = len(self.payload)
        head = bytearray((first,))
        if length < _LENGTH_16:
            head.append(mask_bit | length)
        elif length <= 0xFFFF:
            head.append(mask_bit | _LENGTH_16)
            head += length.to_bytes(2, "big")
        else:
            head.append(mask_bit | _LENGTH_64)
            head += length.to_bytes(8, "big")
        if self.mask_key is None:
            return bytes(head) + self.payload
        head += self.mask_key
        return bytes(head) + apply_mask(self.payload, self.mask_key)


class FrameHeader(NamedTuple):
    """A frame's header, readable before its payload has arrived."""

    fin: bool
    rsv1: bool
    rsv2: bool
    rsv3: bool
    opcode: int
    mask_key: bytes | None
    length: int
    header_size: int
    minimal_length: bool
    """Whether the length used its shortest form and, for the 64-bit
    form, left the top bit clear."""


class DecodedFrame(NamedTuple):
    """A complete frame and the exact bytes that carried it."""

    frame: WebSocketFrame
    wire_raw_bytes: bytes
    header: FrameHeader


def parse_header(data: bytes | bytearray) -> FrameHeader | None:
    """Parse the header at the start of ``data``; ``None`` if short."""
    if len(data) < 2:
        return None
    first, second = data[0], data[1]
    masked = bool(second & 0x80)
    length = second & 0x7F
    offset = 2
    minimal = True
    if length == _LENGTH_16:
        if len(data) < 4:
            return None
        length = int.from_bytes(data[2:4], "big")
        offset = 4
        minimal = length >= _LENGTH_16
    elif length == _LENGTH_64:
        if len(data) < 10:
            return None
        length = int.from_bytes(data[2:10], "big")
        offset = 10
        minimal = 0xFFFF < length < _TOP_BIT_64
    mask_key: bytes | None = None
    if masked:
        if len(data) < offset + 4:
            return None
        mask_key = bytes(data[offset : offset + 4])
        offset += 4
    return FrameHeader(
        fin=bool(first & 0x80),
        rsv1=bool(first & 0x40),
        rsv2=bool(first & 0x20),
        rsv3=bool(first & 0x10),
        opcode=first & 0x0F,
        mask_key=mask_key,
        length=length,
        header_size=offset,
        minimal_length=minimal,
    )


class FrameDecoder:
    """Turn a byte stream into frames, keeping each frame's wire bytes.

    ``feed()`` takes bytes in any split and returns the frames they
    complete.  The decoder holds at most one partial frame, and
    ``partial_header`` exposes its header once that has arrived, so a
    caller can refuse an oversized frame before buffering its payload.
    """

    def __init__(self) -> None:
        self._buffer = bytearray()

    @property
    def buffered(self) -> int:
        """Bytes of the partial frame held so far."""
        return len(self._buffer)

    @property
    def partial_header(self) -> FrameHeader | None:
        """The header of the partial frame, once it is complete."""
        return parse_header(self._buffer)

    def partial_payload(self, start: int = 0) -> bytes:
        """Unmasked payload of the partial frame, from offset ``start``.

        Returns the bytes received so far, so a caller can check a
        long frame as it arrives; unmasking starts at ``start`` rather
        than the beginning, so each call costs only its new bytes.
        """
        header = parse_header(self._buffer)
        if header is None:
            return b""
        data = bytes(self._buffer[header.header_size + start :])
        if header.mask_key is None or not data:
            return data
        shift = start % 4
        key = header.mask_key[shift:] + header.mask_key[:shift]
        return apply_mask(data, key)

    def feed(self, data: bytes) -> list[DecodedFrame]:
        buffer = self._buffer
        buffer += data
        frames: list[DecodedFrame] = []
        while True:
            header = parse_header(buffer)
            if header is None:
                return frames
            total = header.header_size + header.length
            if len(buffer) < total:
                return frames
            wire = bytes(buffer[:total])
            del buffer[:total]
            payload = wire[header.header_size :]
            if header.mask_key is not None:
                payload = apply_mask(payload, header.mask_key)
            opcode = header.opcode
            frame = WebSocketFrame(
                opcode=Opcode(opcode) if opcode in DEFINED_OPCODES else opcode,
                payload=payload,
                fin=header.fin,
                rsv1=header.rsv1,
                rsv2=header.rsv2,
                rsv3=header.rsv3,
                mask_key=header.mask_key,
            )
            frames.append(DecodedFrame(frame, wire, header))
