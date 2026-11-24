from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.websocket.deflate import (
    AgreedExtensions,
    InflateError,
    MessageDeflater,
    MessageInflater,
    MessageTooBig,
    PerMessageDeflate,
    agreed_extensions,
    negotiate,
    parse_extensions,
)

# RFC 7692 section 7.2.3.
HELLO = bytes.fromhex("f248cdc9c90700")
HELLO_SHARED_CONTEXT = bytes.fromhex("f200110000")
HELLO_STORED = bytes.fromhex("000500faff48656c6c6f")
DEFAULTS = PerMessageDeflate()


def _inflater(*, no_context_takeover: bool = False) -> MessageInflater:
    return MessageInflater(
        no_context_takeover=no_context_takeover, max_window_bits=None
    )


def _inflate(inflater: MessageInflater, data: bytes) -> bytes:
    inflater.start()
    return inflater.feed(data, None) + inflater.finish(None)


def test_deflater_matches_rfc_7692_vectors_with_context_takeover() -> None:
    deflater = MessageDeflater(no_context_takeover=False, max_window_bits=None)

    assert deflater.compress(b"Hello") == HELLO
    assert deflater.compress(b"Hello") == HELLO_SHARED_CONTEXT


def test_deflater_without_context_takeover_starts_each_message_fresh() -> None:
    deflater = MessageDeflater(no_context_takeover=True, max_window_bits=None)

    assert deflater.compress(b"Hello") == HELLO
    assert deflater.compress(b"Hello") == HELLO


@pytest.mark.parametrize("payload", [HELLO, HELLO_STORED])
def test_inflater_reads_rfc_7692_vectors(payload: bytes) -> None:
    assert _inflate(_inflater(), payload) == b"Hello"


def test_inflater_keeps_context_across_messages() -> None:
    inflater = _inflater()

    assert _inflate(inflater, HELLO) == b"Hello"
    assert _inflate(inflater, HELLO_SHARED_CONTEXT) == b"Hello"


def test_inflater_without_context_takeover_cannot_read_shared_message() -> (
    None
):
    inflater = _inflater(no_context_takeover=True)
    _inflate(inflater, HELLO)

    with pytest.raises(InflateError):
        _inflate(inflater, HELLO_SHARED_CONTEXT)


def test_inflater_raises_once_message_passes_limit() -> None:
    compressed = MessageDeflater(
        no_context_takeover=True, max_window_bits=None
    ).compress(bytes(100_000))
    inflater = _inflater()
    inflater.start()

    with pytest.raises(MessageTooBig):
        inflater.feed(compressed, 1000)


def test_inflater_accepts_message_exactly_at_limit() -> None:
    compressed = MessageDeflater(
        no_context_takeover=True, max_window_bits=None
    ).compress(bytes(1000))
    inflater = _inflater()
    inflater.start()

    out = inflater.feed(compressed, 1000) + inflater.finish(1000)

    assert out == bytes(1000)


@given(st.lists(st.binary(max_size=2000), min_size=1, max_size=4))
def test_deflate_then_inflate_round_trips_with_context(
    messages: list[bytes],
) -> None:
    deflater = MessageDeflater(no_context_takeover=False, max_window_bits=10)
    inflater = MessageInflater(no_context_takeover=False, max_window_bits=10)

    for message in messages:
        assert _inflate(inflater, deflater.compress(message)) == message


def test_parse_extensions_reads_offers_params_and_quotes() -> None:
    offers = parse_extensions([
        (
            "permessage-deflate; client_max_window_bits; "
            'server_max_window_bits="10"'
        ),
        "x-other, ,Permessage-Deflate",
    ])

    assert offers == [
        (
            "permessage-deflate",
            [
                ("client_max_window_bits", None),
                ("server_max_window_bits", "10"),
            ],
        ),
        ("x-other", []),
        ("permessage-deflate", []),
    ]


def test_negotiate_declines_when_not_offered() -> None:
    assert negotiate(["x-webkit-deflate-frame"], DEFAULTS) is None
    assert negotiate([], DEFAULTS) is None


def test_negotiate_accepts_plain_offer() -> None:
    agreed = negotiate(["permessage-deflate"], DEFAULTS)

    assert agreed == PerMessageDeflate()
    assert agreed.header_value() == "permessage-deflate"


@pytest.mark.parametrize(
    ("offer", "preferences", "expected"),
    [
        (
            "permessage-deflate; server_no_context_takeover",
            DEFAULTS,
            PerMessageDeflate(server_no_context_takeover=True),
        ),
        (
            "permessage-deflate",
            PerMessageDeflate(client_no_context_takeover=True),
            PerMessageDeflate(client_no_context_takeover=True),
        ),
        (
            "permessage-deflate; server_max_window_bits=12",
            PerMessageDeflate(server_max_window_bits=10),
            PerMessageDeflate(server_max_window_bits=10),
        ),
        (
            "permessage-deflate; server_max_window_bits=12",
            DEFAULTS,
            PerMessageDeflate(server_max_window_bits=12),
        ),
        (
            "permessage-deflate; server_max_window_bits=9",
            PerMessageDeflate(server_max_window_bits=12),
            PerMessageDeflate(server_max_window_bits=9),
        ),
        (
            "permessage-deflate",
            PerMessageDeflate(client_max_window_bits=10),
            PerMessageDeflate(),
        ),
        (
            "permessage-deflate; client_max_window_bits",
            PerMessageDeflate(client_max_window_bits=10),
            PerMessageDeflate(client_max_window_bits=10),
        ),
        (
            "permessage-deflate; client_max_window_bits",
            DEFAULTS,
            PerMessageDeflate(),
        ),
        (
            "permessage-deflate; client_max_window_bits=11",
            PerMessageDeflate(client_max_window_bits=13),
            PerMessageDeflate(client_max_window_bits=11),
        ),
    ],
)
def test_negotiate_agrees_parameters(
    offer: str,
    preferences: PerMessageDeflate,
    expected: PerMessageDeflate,
) -> None:
    assert negotiate([offer], preferences) == expected


@pytest.mark.parametrize(
    "skipped",
    [
        "permessage-deflate; unknown_param",
        (
            "permessage-deflate; server_no_context_takeover; "
            "server_no_context_takeover"
        ),
        "permessage-deflate; server_no_context_takeover=1",
        "permessage-deflate; server_max_window_bits",
        "permessage-deflate; server_max_window_bits=16",
        "permessage-deflate; server_max_window_bits=010",
        "permessage-deflate; server_max_window_bits=8",
        "permessage-deflate; client_max_window_bits=7",
    ],
)
def test_negotiate_skips_invalid_offer_for_next_one(skipped: str) -> None:
    agreed = negotiate(
        [f"{skipped}, permessage-deflate; client_no_context_takeover"],
        DEFAULTS,
    )

    assert agreed == PerMessageDeflate(client_no_context_takeover=True)


def test_negotiate_skips_offer_when_preference_forces_window_of_8() -> None:
    assert (
        negotiate(
            ["permessage-deflate"], PerMessageDeflate(server_max_window_bits=8)
        )
        is None
    )


def test_header_value_names_every_agreed_parameter() -> None:
    agreed = PerMessageDeflate(
        server_no_context_takeover=True,
        client_no_context_takeover=True,
        server_max_window_bits=10,
        client_max_window_bits=11,
    )

    assert agreed.header_value() == (
        "permessage-deflate; server_no_context_takeover; "
        "client_no_context_takeover; server_max_window_bits=10; "
        "client_max_window_bits=11"
    )
    assert agreed_extensions([agreed.header_value()]) == AgreedExtensions(
        compression=agreed
    )


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([], AgreedExtensions()),
        (["x-custom"], AgreedExtensions(other=True)),
        (
            ["permessage-deflate; client_max_window_bits"],
            AgreedExtensions(other=True),
        ),
        (
            ["permessage-deflate, permessage-deflate"],
            AgreedExtensions(compression=PerMessageDeflate(), other=True),
        ),
    ],
)
def test_agreed_extensions_reads_response(
    values: list[str], expected: AgreedExtensions
) -> None:
    assert agreed_extensions(values) == expected


@pytest.mark.parametrize("bits", [7, 16])
def test_preferences_reject_window_bits_out_of_range(bits: int) -> None:
    with pytest.raises(ValueError, match="between 8 and 15"):
        PerMessageDeflate(client_max_window_bits=bits)
