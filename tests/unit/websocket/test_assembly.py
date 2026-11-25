from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import pytest

from localstub.websocket.assembly import (
    MessageAssembler,
    Received,
    close_code_allowed,
)
from localstub.websocket.deflate import MessageDeflater, PerMessageDeflate
from localstub.websocket.frames import (
    FrameDecoder,
    FrameHeader,
    Opcode,
    WebSocketFrame,
)
from localstub.websocket.records import (
    RecordedWebSocketFrame,
    Sender,
    WebSocketClose,
    WebSocketMessage,
)

MASK = b"\x01\x02\x03\x04"
TIMESTAMP = datetime(2026, 11, 24, tzinfo=UTC)
DEFLATE = PerMessageDeflate()
SNOWMAN = "\u2603".encode()


def _masked(frame: WebSocketFrame) -> WebSocketFrame:
    return replace(frame, mask_key=MASK)


def _compressed(data: bytes) -> bytes:
    deflater = MessageDeflater(no_context_takeover=True, max_window_bits=None)
    return deflater.compress(data)


def _assembler(
    *,
    sender: Sender = "client",
    strict: bool = True,
    compression: PerMessageDeflate | None = None,
    max_message_bytes: int | None = None,
    unknown_extensions: bool = False,
) -> MessageAssembler:
    return MessageAssembler(
        sender=sender,
        compression=compression,
        strict=strict,
        max_message_bytes=max_message_bytes,
        unknown_extensions=unknown_extensions,
    )


def _feed(
    assembler: MessageAssembler,
    wires: list[bytes],
    sender: Sender = "client",
) -> list[Received]:
    decoder = FrameDecoder()
    results: list[Received] = []
    for wire in wires:
        [decoded] = decoder.feed(wire)
        recorded = RecordedWebSocketFrame(
            frame=decoded.frame,
            sender=sender,
            wire_raw_bytes=decoded.wire_raw_bytes,
            timestamp=TIMESTAMP,
            monotonic=0.0,
        )
        results.append(assembler.receive(recorded, decoded.header))
    return results


def _messages(results: list[Received]) -> list[WebSocketMessage]:
    return [m.message for result in results for m in result.messages]


def _kinds(results: list[Received]) -> list[str]:
    return [v.kind for result in results for v in result.violations]


TEXT = WebSocketMessage("text", b"hi")

# (frames as wire bytes, assembler options, violation kind, close code,
# messages a lenient assembler delivers)
VIOLATIONS = [
    pytest.param(
        [_masked(WebSocketFrame(Opcode.TEXT, SNOWMAN[:2])).encode()],
        {},
        "invalid_utf8",
        1007,
        [WebSocketMessage("text", SNOWMAN[:2])],
        id="text_ends_inside_a_character",
    ),
    pytest.param(
        [WebSocketFrame.text("hi").encode()],
        {},
        "unmasked_frame",
        1002,
        [TEXT],
        id="unmasked_frame",
    ),
    pytest.param(
        [_masked(WebSocketFrame(Opcode.TEXT, b"hi", rsv2=True)).encode()],
        {},
        "reserved_bits",
        1002,
        [TEXT],
        id="reserved_bits",
    ),
    pytest.param(
        [_masked(WebSocketFrame(Opcode.TEXT, b"hi", rsv1=True)).encode()],
        {},
        "reserved_bits",
        1002,
        [TEXT],
        id="rsv1_without_deflate",
    ),
    pytest.param(
        [_masked(WebSocketFrame(0x3, b"x")).encode()],
        {},
        "reserved_opcode",
        1002,
        [],
        id="reserved_opcode_data",
    ),
    pytest.param(
        [_masked(WebSocketFrame(0xB)).encode()],
        {},
        "reserved_opcode",
        1002,
        [],
        id="reserved_opcode_control",
    ),
    pytest.param(
        [_masked(WebSocketFrame(Opcode.PING, fin=False)).encode()],
        {},
        "fragmented_control_frame",
        1002,
        [],
        id="fragmented_control_frame",
    ),
    pytest.param(
        [_masked(WebSocketFrame.ping(bytes(126))).encode()],
        {},
        "oversized_control_frame",
        1002,
        [],
        id="oversized_control_frame",
    ),
    pytest.param(
        [_masked(WebSocketFrame.continuation(b"hi")).encode()],
        {},
        "unexpected_continuation",
        1002,
        [WebSocketMessage("binary", b"hi")],
        id="unexpected_continuation",
    ),
    pytest.param(
        [
            _masked(WebSocketFrame.text("a", fin=False)).encode(),
            _masked(WebSocketFrame.text("hi")).encode(),
        ],
        {},
        "interrupted_message",
        1002,
        [WebSocketMessage("text", b"a"), TEXT],
        id="interrupted_message",
    ),
    pytest.param(
        [b"\x81\xfe\x00\x02" + MASK + bytes([0x69, 0x6B])],
        {},
        "invalid_length",
        1002,
        [TEXT],
        id="invalid_length",
    ),
    pytest.param(
        [_masked(WebSocketFrame(Opcode.CLOSE, b"\x03")).encode()],
        {},
        "invalid_close",
        1002,
        [],
        id="close_payload_of_one_byte",
    ),
    pytest.param(
        [_masked(WebSocketFrame.close(1005)).encode()],
        {},
        "invalid_close",
        1002,
        [],
        id="close_code_not_allowed",
    ),
    pytest.param(
        [
            _masked(WebSocketFrame.close(1000)).encode(),
            _masked(WebSocketFrame.text("hi")).encode(),
        ],
        {},
        "frame_after_close",
        1002,
        [TEXT],
        id="frame_after_close",
    ),
    pytest.param(
        [_masked(WebSocketFrame(Opcode.TEXT, b"\xff")).encode()],
        {},
        "invalid_utf8",
        1007,
        [WebSocketMessage("text", b"\xff")],
        id="invalid_utf8_text",
    ),
    pytest.param(
        [_masked(WebSocketFrame(Opcode.CLOSE, b"\x03\xe8\xff")).encode()],
        {},
        "invalid_utf8",
        1007,
        [],
        id="invalid_utf8_close_reason",
    ),
    pytest.param(
        [
            _masked(
                WebSocketFrame(Opcode.BINARY, b"\xff\xff\xff", rsv1=True)
            ).encode()
        ],
        {"compression": DEFLATE},
        "invalid_compressed_data",
        1007,
        [WebSocketMessage("binary", b"\xff\xff\xff")],
        id="invalid_compressed_data",
    ),
    pytest.param(
        # A stored block whose lengths only disagree once the sync tail
        # completes them: the failure surfaces as the message ends.
        [
            _masked(
                WebSocketFrame(Opcode.BINARY, b"\x00\x01", rsv1=True)
            ).encode()
        ],
        {"compression": DEFLATE},
        "invalid_compressed_data",
        1007,
        [WebSocketMessage("binary", b"\x00\x01")],
        id="invalid_compressed_data_at_message_end",
    ),
    pytest.param(
        [_masked(WebSocketFrame.text("hello")).encode()],
        {"max_message_bytes": 4},
        "message_too_big",
        1009,
        [WebSocketMessage("text", b"hello")],
        id="message_too_big",
    ),
    pytest.param(
        [
            _masked(
                WebSocketFrame(
                    Opcode.BINARY, _compressed(bytes(100)), rsv1=True
                )
            ).encode()
        ],
        {"compression": DEFLATE, "max_message_bytes": 50},
        "message_too_big",
        1009,
        [WebSocketMessage("binary", bytes(100))],
        id="inflated_message_too_big",
    ),
]


@pytest.mark.parametrize(
    ("wires", "options", "kind", "close_code", "lenient_messages"),
    VIOLATIONS,
)
def test_strict_assembler_records_violation_and_stops(
    wires: list[bytes],
    options: dict[str, Any],
    kind: str,
    close_code: int,
    lenient_messages: list[WebSocketMessage],
) -> None:
    _ = lenient_messages
    assembler = _assembler(strict=True, **options)

    results = _feed(assembler, wires)

    violations = [v for result in results for v in result.violations]
    assert [v.kind for v in violations] == [kind]
    assert violations[0].close_code == close_code
    assert violations[0].sender == "client"
    assert violations[0].frame is not None
    later = _feed(assembler, [_masked(WebSocketFrame.text("x")).encode()])
    assert later == [Received()]


@pytest.mark.parametrize(
    ("wires", "options", "kind", "close_code", "lenient_messages"),
    VIOLATIONS,
)
def test_lenient_assembler_records_violation_and_delivers(
    wires: list[bytes],
    options: dict[str, Any],
    kind: str,
    close_code: int,
    lenient_messages: list[WebSocketMessage],
) -> None:
    _ = close_code
    assembler = _assembler(strict=False, **options)

    results = _feed(assembler, wires)

    assert _kinds(results) == [kind]
    assert _messages(results) == lenient_messages
    later = _feed(assembler, [_masked(WebSocketFrame.binary(b"x")).encode()])
    assert _messages(later) == [WebSocketMessage("binary", b"x")]


def test_fragmented_text_reassembles_with_its_frames() -> None:
    assembler = _assembler()

    results = _feed(
        assembler,
        [
            _masked(WebSocketFrame.text("Hel", fin=False)).encode(),
            _masked(WebSocketFrame.ping(b"p")).encode(),
            _masked(WebSocketFrame.continuation(b"lo")).encode(),
        ],
    )

    assert _kinds(results) == []
    [message] = results[2].messages
    assert message.message == WebSocketMessage("text", b"Hello")
    assert [f.frame.payload for f in message.frames] == [b"Hel", b"lo"]
    assert not message.compressed


def test_utf8_split_across_fragments_is_valid() -> None:
    snowman = "\u2603".encode()

    results = _feed(
        _assembler(),
        [
            _masked(
                WebSocketFrame(Opcode.TEXT, snowman[:1], fin=False)
            ).encode(),
            _masked(WebSocketFrame.continuation(snowman[1:])).encode(),
        ],
    )

    assert _kinds(results) == []
    assert _messages(results) == [WebSocketMessage("text", snowman)]


def test_compressed_fragmented_message_inflates() -> None:
    payload = _compressed(b"hello hello hello")
    assembler = _assembler(compression=DEFLATE)

    results = _feed(
        assembler,
        [
            _masked(
                WebSocketFrame(Opcode.TEXT, payload[:4], fin=False, rsv1=True)
            ).encode(),
            _masked(WebSocketFrame.continuation(payload[4:])).encode(),
        ],
    )

    [message] = results[1].messages
    assert message.compressed
    assert message.message == WebSocketMessage("text", b"hello hello hello")
    assert b"".join(f.frame.payload for f in message.frames) == payload


def test_uncompressed_message_on_deflate_session_is_not_inflated() -> None:
    results = _feed(
        _assembler(compression=DEFLATE),
        [_masked(WebSocketFrame.text("plain")).encode()],
    )

    [message] = results[0].messages
    assert not message.compressed
    assert message.data == b"plain"


def test_rsv1_on_continuation_is_reserved_even_with_deflate() -> None:
    results = _feed(
        _assembler(compression=DEFLATE),
        [
            _masked(WebSocketFrame.text("a", fin=False)).encode(),
            _masked(
                WebSocketFrame(Opcode.CONTINUATION, b"b", rsv1=True)
            ).encode(),
        ],
    )

    assert _kinds(results) == ["reserved_bits"]


def test_server_sender_must_not_mask() -> None:
    assembler = _assembler(sender="server", strict=False)

    results = _feed(
        assembler,
        [
            WebSocketFrame.text("ok").encode(),
            _masked(WebSocketFrame.text("hi")).encode(),
        ],
        sender="server",
    )

    assert _kinds(results) == ["masked_frame"]
    assert results[1].violations[0].sender == "server"
    assert _messages(results) == [WebSocketMessage("text", b"ok"), TEXT]


def test_server_sender_inflates_with_server_parameters() -> None:
    deflater = MessageDeflater(no_context_takeover=False, max_window_bits=9)
    agreed = PerMessageDeflate(server_max_window_bits=9)
    assembler = _assembler(sender="server", compression=agreed)

    results = _feed(
        assembler,
        [
            WebSocketFrame(
                Opcode.BINARY, deflater.compress(bytes(600)), rsv1=True
            ).encode(),
            WebSocketFrame(
                Opcode.BINARY, deflater.compress(bytes(600)), rsv1=True
            ).encode(),
        ],
        sender="server",
    )

    assert _messages(results) == [WebSocketMessage("binary", bytes(600))] * 2


def test_client_no_context_takeover_resets_each_message() -> None:
    deflater = MessageDeflater(no_context_takeover=True, max_window_bits=None)
    agreed = PerMessageDeflate(client_no_context_takeover=True)
    frames = [
        _masked(
            WebSocketFrame(Opcode.TEXT, deflater.compress(b"same"), rsv1=True)
        ).encode()
        for _ in range(2)
    ]

    results = _feed(_assembler(compression=agreed), frames)

    assert _messages(results) == [WebSocketMessage("text", b"same")] * 2


def test_close_frame_reports_code_and_reason() -> None:
    [result] = _feed(
        _assembler(), [_masked(WebSocketFrame.close(4000, "bye")).encode()]
    )

    assert result.close == WebSocketClose(4000, "bye")
    assert result.violations == ()


def test_empty_close_frame_reports_no_code() -> None:
    [result] = _feed(
        _assembler(), [_masked(WebSocketFrame.close(None)).encode()]
    )

    assert result.close == WebSocketClose(None)


def test_unknown_extensions_skip_reserved_bits_and_utf8_checks() -> None:
    assembler = _assembler(unknown_extensions=True)

    results = _feed(
        assembler,
        [
            _masked(
                WebSocketFrame(Opcode.TEXT, b"\xff", rsv1=True, rsv3=True)
            ).encode()
        ],
    )

    assert _kinds(results) == []
    assert _messages(results) == [WebSocketMessage("text", b"\xff")]


def _header(wire: bytes) -> FrameHeader | None:
    decoder = FrameDecoder()
    decoder.feed(wire)
    return decoder.partial_header


@pytest.mark.parametrize(
    ("wire", "options", "kind"),
    [
        (b"\x82\xfe\x00\x05" + MASK, {}, "invalid_length"),
        (b"\x89\xfe\x00\x80" + MASK, {}, "oversized_control_frame"),
        (
            b"\x82\xfe\x01\x00" + MASK,
            {"max_message_bytes": 255},
            "message_too_big",
        ),
    ],
)
def test_check_header_refuses_frame_before_payload(
    wire: bytes, options: dict[str, Any], kind: str
) -> None:
    assembler = _assembler(**options)
    header = _header(wire)
    assert header is not None

    violation = assembler.check_header(header)

    assert violation is not None
    assert violation.kind == kind
    assert violation.frame is None
    assert assembler.check_header(header) is None


@pytest.mark.parametrize(
    ("earlier", "wire", "kind"),
    [
        pytest.param(
            WebSocketFrame.binary(bytes(60)),
            b"\x82\xbc" + MASK,
            None,
            id="new_message",
        ),
        pytest.param(
            WebSocketFrame(Opcode.BINARY, bytes(60), fin=False),
            b"\x80\xbc" + MASK,
            "message_too_big",
            id="continuation",
        ),
    ],
)
def test_check_header_counts_only_the_message_in_progress(
    earlier: WebSocketFrame, wire: bytes, kind: str | None
) -> None:
    assembler = _assembler(max_message_bytes=100)
    _feed(assembler, [_masked(earlier).encode()])
    header = _header(wire)
    assert header is not None

    violation = assembler.check_header(header)

    assert (None if violation is None else violation.kind) == kind


@pytest.mark.parametrize(
    ("wire", "options"),
    [
        (b"\x82\xfe\x01\x00" + MASK, {"max_message_bytes": 256}),
        (b"\x89\x7d" + MASK, {}),
        (
            b"\xc2\xfe\x01\x00" + MASK,
            {"max_message_bytes": 10, "compression": DEFLATE},
        ),
    ],
)
def test_check_header_allows_frame_within_limits(
    wire: bytes, options: dict[str, Any]
) -> None:
    header = _header(wire)
    assert header is not None

    assert _assembler(**options).check_header(header) is None


@pytest.mark.parametrize(
    ("code", "allowed"),
    [
        (999, False),
        (1000, True),
        (1003, True),
        (1004, False),
        (1006, False),
        (1007, True),
        (1014, True),
        (1015, False),
        (2999, False),
        (3000, True),
        (4999, True),
        (5000, False),
    ],
)
def test_close_code_allowed_follows_rfc_6455(code: int, allowed: bool) -> None:
    assert close_code_allowed(code) == allowed


def test_invalid_utf8_fails_at_its_fragment_not_at_fin() -> None:
    results = _feed(
        _assembler(),
        [
            _masked(WebSocketFrame.text("ok", fin=False)).encode(),
            _masked(WebSocketFrame.continuation(b"\xff", fin=False)).encode(),
        ],
    )

    assert [len(result.violations) for result in results] == [0, 1]
    assert results[1].violations[0].kind == "invalid_utf8"


def _partial(
    assembler: MessageAssembler, wire: bytes, chunks: list[int]
) -> list[str | None]:
    """Feed ``wire`` cut at ``chunks``, checking the partial frame."""
    decoder = FrameDecoder()
    kinds: list[str | None] = []
    start = 0
    for end in chunks:
        assert decoder.feed(wire[start:end]) == []
        start = end
        header = decoder.partial_header
        assert header is not None
        violation = assembler.check_partial(header, decoder)
        kinds.append(None if violation is None else violation.kind)
    return kinds


def test_check_partial_refuses_invalid_utf8_before_frame_ends() -> None:
    payload = SNOWMAN + b"\xff" + bytes(20)
    wire = _masked(WebSocketFrame(Opcode.TEXT, payload)).encode()
    # Six header bytes; the snowman arrives split, then the bad byte.
    assembler = _assembler()

    kinds = _partial(assembler, wire, [7, 9, 10, 12])

    assert kinds == [None, None, "invalid_utf8", None]


@pytest.mark.parametrize(
    ("rest", "kind"),
    [(SNOWMAN[1:] + b"ab", None), (b"ab", "invalid_utf8")],
)
def test_check_partial_continues_text_from_earlier_fragment(
    rest: bytes, kind: str | None
) -> None:
    assembler = _assembler()
    first = WebSocketFrame(Opcode.TEXT, SNOWMAN[:1], fin=False)
    _feed(assembler, [_masked(first).encode()])
    wire = _masked(WebSocketFrame.continuation(rest)).encode()

    assert _partial(assembler, wire, [len(wire) - 1]) == [kind]


BAD = b"\xff" * 10


@pytest.mark.parametrize(
    ("earlier", "frame", "options"),
    [
        pytest.param(
            [], WebSocketFrame(Opcode.BINARY, BAD), {}, id="binary_frame"
        ),
        pytest.param(
            [WebSocketFrame(Opcode.BINARY, b"", fin=False)],
            WebSocketFrame.continuation(BAD),
            {},
            id="binary_continuation",
        ),
        pytest.param(
            [],
            WebSocketFrame(Opcode.TEXT, BAD, rsv1=True),
            {"compression": DEFLATE},
            id="compressed_text_frame",
        ),
        pytest.param(
            [
                WebSocketFrame(
                    Opcode.TEXT, _compressed(b"x" * 50), fin=False, rsv1=True
                )
            ],
            WebSocketFrame.continuation(BAD),
            {"compression": DEFLATE},
            id="compressed_continuation",
        ),
        pytest.param(
            [],
            WebSocketFrame(Opcode.TEXT, BAD),
            {"unknown_extensions": True},
            id="unknown_extension",
        ),
    ],
)
def test_check_partial_skips_frames_it_cannot_judge(
    earlier: list[WebSocketFrame],
    frame: WebSocketFrame,
    options: dict[str, Any],
) -> None:
    assembler = _assembler(**options)
    _feed(assembler, [_masked(f).encode() for f in earlier])
    wire = _masked(frame).encode()

    assert _partial(assembler, wire, [len(wire) - 1]) == [None]


def test_check_partial_after_failure_reports_nothing() -> None:
    assembler = _assembler()
    _feed(assembler, [WebSocketFrame.text("unmasked").encode()])
    wire = _masked(WebSocketFrame(Opcode.TEXT, BAD)).encode()

    assert _partial(assembler, wire, [len(wire) - 1]) == [None]
