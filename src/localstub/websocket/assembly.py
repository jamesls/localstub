"""Frames in, messages and violations out, for one sender.

``MessageAssembler`` applies RFC 6455's rules to the frames one peer
sends, reassembles fragmented data frames into messages, and inflates
permessage-deflate messages.  Stub sessions and the forward relay's
taps share it, so both apply the same checks and produce the same
records.

In strict mode the assembler stops at the first violation, since the
caller then fails the connection.  In lenient mode it records every
violation and keeps going, delivering payloads as received.
"""

from __future__ import annotations

import codecs
from typing import NamedTuple

from localstub.websocket.deflate import (
    InflateError,
    MessageInflater,
    MessageTooBig,
    PerMessageDeflate,
)
from localstub.websocket.frames import (
    DEFINED_OPCODES,
    FrameDecoder,
    FrameHeader,
    Opcode,
)
from localstub.websocket.records import (
    MessageKind,
    RecordedWebSocketFrame,
    RecordedWebSocketMessage,
    Sender,
    ViolationKind,
    WebSocketClose,
    WebSocketMessage,
    WebSocketViolation,
)

MAX_CONTROL_PAYLOAD = 125
_UTF8 = codecs.getincrementaldecoder("utf-8")

_CLOSE_CODES: dict[ViolationKind, int] = {
    "unmasked_frame": 1002,
    "masked_frame": 1002,
    "reserved_bits": 1002,
    "reserved_opcode": 1002,
    "fragmented_control_frame": 1002,
    "oversized_control_frame": 1002,
    "unexpected_continuation": 1002,
    "interrupted_message": 1002,
    "invalid_length": 1002,
    "invalid_close": 1002,
    "frame_after_close": 1002,
    "invalid_utf8": 1007,
    "invalid_compressed_data": 1007,
    "message_too_big": 1009,
}

_DESCRIPTIONS: dict[ViolationKind, str] = {
    "unmasked_frame": "client frame is not masked",
    "masked_frame": "server frame is masked",
    "reserved_bits": "RSV bit set that no negotiated extension defines",
    "reserved_opcode": "reserved opcode",
    "fragmented_control_frame": "control frame with FIN clear",
    "oversized_control_frame": "control frame payload over 125 bytes",
    "unexpected_continuation": "continuation with no message in progress",
    "interrupted_message": "new data frame while a message is in progress",
    "invalid_length": "payload length not in its shortest form",
    "invalid_close": "invalid Close frame payload or code",
    "frame_after_close": "data frame after Close",
    "invalid_utf8": "text that is not UTF-8",
    "invalid_compressed_data": "compressed message does not inflate",
    "message_too_big": "message larger than max_message_bytes",
}


def close_code_allowed(code: int) -> bool:
    """Whether ``code`` may appear in a Close frame (RFC 6455 §7.4).

    1004 to 1006 and 1015 are reserved, 1016 to 2999 are unassigned,
    and 1012 to 1014 are registered with IANA.
    """
    return 1000 <= code <= 1003 or 1007 <= code <= 1014 or 3000 <= code <= 4999


class Received(NamedTuple):
    """What one frame produced."""

    violations: tuple[WebSocketViolation, ...] = ()
    messages: tuple[RecordedWebSocketMessage, ...] = ()
    close: WebSocketClose | None = None


class MessageAssembler:
    """Check and reassemble the frames one sender sends."""

    def __init__(
        self,
        *,
        sender: Sender,
        compression: PerMessageDeflate | None,
        strict: bool,
        max_message_bytes: int | None,
        unknown_extensions: bool = False,
    ) -> None:
        """Set ``unknown_extensions`` when the handshake agreed an
        extension other than permessage-deflate: its use of the RSV
        bits and of the payload cannot be judged, so reserved bits and
        UTF-8 are not checked and payloads are kept uninterpreted.
        """
        self._sender = sender
        self._unknown_extensions = unknown_extensions
        self._strict = strict
        self._limit = max_message_bytes
        self._inflater: MessageInflater | None = None
        if compression is not None:
            if sender == "client":
                self._inflater = MessageInflater(
                    no_context_takeover=compression.client_no_context_takeover,
                    max_window_bits=compression.client_max_window_bits,
                )
            else:
                self._inflater = MessageInflater(
                    no_context_takeover=compression.server_no_context_takeover,
                    max_window_bits=compression.server_max_window_bits,
                )
        self._failed = False
        self._closed = False
        self._found: list[WebSocketViolation] = []
        self._last_close: WebSocketClose | None = None
        self._kind: MessageKind | None = None
        self._compressed = False
        self._inflate_ok = True
        self._frames: list[RecordedWebSocketFrame] = []
        self._chunks: list[bytes] = []
        self._raw: list[bytes] = []
        self._size = 0
        self._size_reported = False
        # Validates a text message as its payload arrives, so invalid
        # UTF-8 fails the connection at the first bad byte.
        self._utf8: codecs.IncrementalDecoder | None = None
        # The same check for the partial frame still arriving; ``None``
        # with ``_partial_known`` set means the frame is not text.
        self._partial_utf8: codecs.IncrementalDecoder | None = None
        self._partial_checked = 0
        self._partial_known = False

    def release(self) -> None:
        """Drop the decompression context once the session has ended."""
        self._inflater = None
        self._utf8 = None
        self._partial_utf8 = None

    def check_partial(
        self, header: FrameHeader, decoder: FrameDecoder
    ) -> WebSocketViolation | None:
        """Refuse a text frame at its first invalid UTF-8 byte.

        Checks the payload of the partial frame received so far, for an
        uncompressed text frame or a continuation of one.  Only the
        bytes that arrived since the last call are decoded.
        """
        if self._failed:
            return None
        if not self._partial_known:
            self._partial_known = True
            self._partial_utf8 = self._partial_decoder(header)
            self._partial_checked = 0
        utf8 = self._partial_utf8
        if utf8 is None:
            return None
        data = decoder.partial_payload(self._partial_checked)
        self._partial_checked += len(data)
        try:
            utf8.decode(data)
        except UnicodeDecodeError:
            return self._header_violation("invalid_utf8")
        return None

    def _partial_decoder(
        self, header: FrameHeader
    ) -> codecs.IncrementalDecoder | None:
        if self._unknown_extensions or (
            header.rsv1 and self._inflater is not None
        ):
            return None
        if header.opcode == Opcode.TEXT and self._kind is None:
            return _UTF8()
        utf8 = self._utf8
        if header.opcode != Opcode.CONTINUATION or utf8 is None:
            return None
        if self._compressed:
            return None
        copy = _UTF8()
        copy.setstate(utf8.getstate())
        return copy

    def check_header(self, header: FrameHeader) -> WebSocketViolation | None:
        """Refuse a frame from its header, before its payload arrives.

        Used in strict mode for frames long enough that buffering them
        before failing would cost memory.  Returns the violation, which
        the caller fails the connection with, or ``None``.
        """
        if self._failed:
            return None
        if not header.minimal_length:
            return self._header_violation("invalid_length")
        if header.opcode >= Opcode.CLOSE:
            if header.length > MAX_CONTROL_PAYLOAD:
                return self._header_violation("oversized_control_frame")
            return None
        compressed = header.rsv1 and self._inflater is not None
        # Only a continuation adds to the message in progress.
        size = self._size if header.opcode == Opcode.CONTINUATION else 0
        if (
            not compressed
            and self._limit is not None
            and size + header.length > self._limit
        ):
            return self._header_violation("message_too_big")
        return None

    def _header_violation(self, kind: ViolationKind) -> WebSocketViolation:
        self._failed = True
        return WebSocketViolation(
            kind, _CLOSE_CODES[kind], self._sender, _DESCRIPTIONS[kind]
        )

    def receive(
        self,
        frame: RecordedWebSocketFrame,
        header: FrameHeader,
    ) -> Received:
        """Apply one complete frame."""
        if self._failed:
            return Received()
        self._found = []
        self._last_close = None
        self._partial_known = False
        self._partial_utf8 = None
        messages: list[RecordedWebSocketMessage] = []
        self._receive(frame, header, messages)
        return Received(tuple(self._found), tuple(messages), self._last_close)

    def _receive(
        self,
        recorded: RecordedWebSocketFrame,
        header: FrameHeader,
        messages: list[RecordedWebSocketMessage],
    ) -> None:
        frame = recorded.frame
        masked = frame.mask_key is not None
        if masked != (self._sender == "client") and self._violate(
            "masked_frame" if masked else "unmasked_frame", recorded
        ):
            return
        if not header.minimal_length and self._violate(
            "invalid_length", recorded
        ):
            return
        if frame.opcode not in DEFINED_OPCODES:
            # A reserved opcode has no meaning to deliver, so the frame
            # is dropped in either mode once the violation is recorded.
            self._violate(
                "reserved_opcode",
                recorded,
                f"reserved opcode 0x{frame.opcode:x}",
            )
            return
        rsv1_allowed = (
            frame.opcode in {Opcode.TEXT, Opcode.BINARY}
            and self._inflater is not None
        )
        if (
            not self._unknown_extensions
            and (frame.rsv2 or frame.rsv3 or (frame.rsv1 and not rsv1_allowed))
            and self._violate("reserved_bits", recorded)
        ):
            return
        if frame.opcode >= Opcode.CLOSE:
            self._control(recorded)
            return
        self._data(recorded, messages)

    def _control(self, recorded: RecordedWebSocketFrame) -> None:
        frame = recorded.frame
        if not frame.fin and self._violate(
            "fragmented_control_frame", recorded
        ):
            return
        if len(frame.payload) > MAX_CONTROL_PAYLOAD and self._violate(
            "oversized_control_frame", recorded
        ):
            return
        if frame.opcode == Opcode.CLOSE:
            self._closed = True
            self._last_close = self._parse_close(recorded)

    def _parse_close(self, recorded: RecordedWebSocketFrame) -> WebSocketClose:
        payload = recorded.frame.payload
        if not payload:
            return WebSocketClose(None)
        if len(payload) == 1:
            self._violate(
                "invalid_close", recorded, "Close payload of one byte"
            )
            return WebSocketClose(None)
        code = int.from_bytes(payload[:2], "big")
        raw_reason = payload[2:]
        if not close_code_allowed(code) and self._violate(
            "invalid_close", recorded, f"close code {code} must not be sent"
        ):
            return WebSocketClose(code, raw_reason.decode(errors="replace"))
        try:
            reason = raw_reason.decode("utf-8")
        except UnicodeDecodeError:
            self._violate(
                "invalid_utf8", recorded, "close reason is not UTF-8"
            )
            reason = raw_reason.decode("utf-8", errors="replace")
        return WebSocketClose(code, reason)

    def _data(
        self,
        recorded: RecordedWebSocketFrame,
        messages: list[RecordedWebSocketMessage],
    ) -> None:
        frame = recorded.frame
        if self._closed and self._violate("frame_after_close", recorded):
            return
        if frame.opcode == Opcode.CONTINUATION:
            if self._kind is None:
                if self._violate("unexpected_continuation", recorded):
                    return
                self._begin("binary", recorded)
            stop = self._add(recorded)
        else:
            if self._kind is not None:
                if self._violate("interrupted_message", recorded):
                    return
                # Only a lenient assembler gets here, and its _finish()
                # always completes the message.
                message = self._finish()
                assert message is not None
                messages.append(message)
            self._begin(
                "text" if frame.opcode == Opcode.TEXT else "binary", recorded
            )
            stop = self._add(recorded)
        if stop or not frame.fin:
            return
        message = self._finish()
        if message is not None:
            messages.append(message)

    def _begin(
        self, kind: MessageKind, recorded: RecordedWebSocketFrame
    ) -> None:
        self._kind = kind
        self._compressed = recorded.frame.rsv1 and self._inflater is not None
        self._inflate_ok = True
        self._frames = []
        self._chunks = []
        self._raw = []
        self._size = 0
        self._size_reported = False
        self._utf8 = (
            _UTF8()
            if kind == "text" and not self._unknown_extensions
            else None
        )
        if self._compressed and self._inflater is not None:
            self._inflater.start()

    def _inflate_limit(self) -> int | None:
        # Lenient mode inflates past the limit so the message can still
        # be delivered; the size check below records the violation.
        return self._limit if self._strict else None

    def _add(self, recorded: RecordedWebSocketFrame) -> bool:
        """Append a data frame's payload; return whether to stop."""
        payload = recorded.frame.payload
        self._frames.append(recorded)
        if self._compressed:
            self._raw.append(payload)
            return self._inflate(recorded, payload, last=False)
        self._chunks.append(payload)
        self._size += len(payload)
        return self._check_size(recorded) or self._check_text(
            recorded, payload, final=False
        )

    def _inflate(
        self,
        recorded: RecordedWebSocketFrame,
        data: bytes,
        *,
        last: bool,
    ) -> bool:
        inflater = self._inflater
        if not self._inflate_ok or inflater is None:
            return False
        try:
            if last:
                out = inflater.finish(self._inflate_limit())
            else:
                out = inflater.feed(data, self._inflate_limit())
        except MessageTooBig:
            return self._violate("message_too_big", recorded)
        except InflateError as exc:
            # The raw payload that is delivered instead is not text.
            self._inflate_ok = False
            self._utf8 = None
            return self._violate(
                "invalid_compressed_data",
                recorded,
                f"compressed message does not inflate: {exc}",
            )
        self._chunks.append(out)
        self._size += len(out)
        return self._check_size(recorded) or self._check_text(
            recorded, out, final=last
        )

    def _check_text(
        self,
        recorded: RecordedWebSocketFrame,
        data: bytes,
        *,
        final: bool,
    ) -> bool:
        """Validate the next text bytes; return whether to stop.

        After the first violation a lenient assembler stops checking,
        so one message records it once.
        """
        utf8 = self._utf8
        if utf8 is None:
            return False
        try:
            utf8.decode(data, final)
        except UnicodeDecodeError:
            self._utf8 = None
            return self._violate("invalid_utf8", recorded)
        return False

    def _check_size(self, recorded: RecordedWebSocketFrame) -> bool:
        if (
            self._limit is None
            or self._size <= self._limit
            or self._size_reported
        ):
            return False
        self._size_reported = True
        return self._violate("message_too_big", recorded)

    def _finish(self) -> RecordedWebSocketMessage | None:
        """Complete the message in progress; ``None`` means stop."""
        kind = self._kind
        assert kind is not None
        last = self._frames[-1]
        self._kind = None
        if self._compressed and self._inflate(last, b"", last=True):
            return None
        inflated = self._compressed and self._inflate_ok
        if self._compressed and not inflated:
            data = b"".join(self._raw)
        else:
            data = b"".join(self._chunks)
        if not self._compressed and self._check_text(last, b"", final=True):
            return None
        return RecordedWebSocketMessage(
            message=WebSocketMessage(kind, data),
            sender=self._sender,
            frames=tuple(self._frames),
            compressed=inflated,
            timestamp=last.timestamp,
        )

    def _violate(
        self,
        kind: ViolationKind,
        recorded: RecordedWebSocketFrame,
        description: str | None = None,
    ) -> bool:
        """Record a violation; return whether to stop (strict mode)."""
        self._found.append(
            WebSocketViolation(
                kind=kind,
                close_code=_CLOSE_CODES[kind],
                sender=self._sender,
                description=description or _DESCRIPTIONS[kind],
                frame=recorded,
            )
        )
        if self._strict:
            self._failed = True
        return self._strict
