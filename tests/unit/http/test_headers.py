from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from localstub.http.headers import HeaderItem, Headers
from tests.unit.http.strategies import obs_text_value


def test_headers_are_immutable() -> None:
    headers = Headers.from_items([("X-Test", "a")])

    assert headers["X-Test"] == "a"

    with pytest.raises(TypeError):
        headers["X-Test"] = "b"
    with pytest.raises(TypeError):
        del headers["X-Test"]


def test_headers_patch_set_overrides_without_mutating_original() -> None:
    headers = Headers.from_items([("X-Test", "a")])
    patched = headers.patch_set({"X-Test": "b", "X-Other": "c"})

    assert headers["X-Test"] == "a"
    assert patched["X-Test"] == "b"
    assert patched["X-Other"] == "c"


def test_headers_equal_when_items_match_despite_patch_history() -> None:
    built = Headers.from_items([("X-Other", "c"), ("X-Test", "b")])
    patched = Headers.from_items([("X-Test", "a"), ("X-Other", "c")])
    patched = patched.patch_set({"X-Test": "b"})

    assert list(built.items()) == list(patched.items())
    assert built == patched
    assert hash(built) == hash(patched)


def test_headers_unequal_when_items_differ() -> None:
    first = Headers.from_items([("X-Test", "a")])
    second = Headers.from_items([("X-Test", "b")])

    assert first != second


def test_headers_order_is_part_of_equality() -> None:
    first = Headers.from_items([("A", "1"), ("B", "2")])
    second = Headers.from_items([("B", "2"), ("A", "1")])

    assert first != second


def test_headers_not_equal_to_other_types() -> None:
    headers = Headers.from_items([("X-Test", "a")])

    assert headers != [("X-Test", "a")]


@given(values=st.lists(obs_text_value(), min_size=1, max_size=4))
def test_headers_obs_text_raw_and_string_views_round_trip(
    values: list[bytes],
) -> None:
    raw = tuple((b"X-Obs", value) for value in values)
    headers = Headers.from_raw_items(raw)
    decoded = [value.decode("latin-1") for value in values]

    assert headers.raw == raw
    assert headers.get_all("x-obs") == decoded
    assert headers["X-OBS"] == decoded[0]
    assert list(headers.items()) == [("X-Obs", value) for value in decoded]
    assert Headers.from_items(headers.items()).raw == raw


@given(original=obs_text_value(), replacement=obs_text_value())
def test_headers_patch_set_preserves_obs_text_bytes(
    original: bytes,
    replacement: bytes,
) -> None:
    headers = Headers.from_raw_items([
        (b"X-Obs", original),
        (b"X-Keep", b"unchanged"),
        (b"x-obs", original),
    ])

    patched = headers.patch_set({
        "x-OBS": replacement.decode("latin-1"),
    })

    assert headers.raw == (
        (b"X-Obs", original),
        (b"X-Keep", b"unchanged"),
        (b"x-obs", original),
    )
    assert patched.raw == (
        (b"X-Keep", b"unchanged"),
        (b"x-OBS", replacement),
    )
    assert patched["X-Obs"] == replacement.decode("latin-1")


def test_headers_get_with_non_ascii_name_returns_default() -> None:
    headers = Headers.from_items([("X-Test", "a")])

    assert headers.get("\u00e9") is None
    assert headers.get("\u00e9", "fallback") == "fallback"


def test_headers_get_all_with_non_ascii_name_returns_failobj() -> None:
    headers = Headers.from_items([("X-Test", "a")])

    assert headers.get_all("\u00e9") is None
    assert headers.get_all("\u00e9", []) == []


def test_headers_contains_with_non_ascii_name_is_false() -> None:
    headers = Headers.from_items([("X-Test", "a")])

    assert "\u00e9" not in headers


def test_headers_getitem_with_non_ascii_name_raises_key_error() -> None:
    headers = Headers.from_items([("X-Test", "a")])

    with pytest.raises(KeyError):
        headers["\u00e9"]


def test_headers_patched_lookup_with_non_ascii_name_returns_default() -> None:
    headers = Headers.from_items([("X-Test", "a")])
    patched = headers.patch_set({"X-Other": "b"})

    assert patched.get("\u00e9") is None
    assert "\u00e9" not in patched


def test_headers_from_items_with_non_ascii_name_raises_value_error() -> None:
    with pytest.raises(ValueError, match="header name must be ASCII"):
        Headers.from_items([("X-\u00e9", "a")])


def test_headers_from_items_with_non_latin1_value_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Latin-1 range"):
        Headers.from_items([("X-Test", "\u2192")])


def test_headers_from_raw_items_with_non_ascii_name_raises_value_error() -> (
    None
):
    with pytest.raises(ValueError, match="header name must be ASCII"):
        Headers.from_raw_items([(b"X-\xe9", b"a")])


def test_headers_from_raw_items_accepts_any_value_bytes() -> None:
    headers = Headers.from_raw_items([(b"X-Test", bytes(range(0x100)))])

    assert headers.raw == ((b"X-Test", bytes(range(0x100))),)


def test_headers_patch_set_with_non_ascii_name_raises_value_error() -> None:
    headers = Headers.from_items([("X-Test", "a")])

    with pytest.raises(ValueError, match="header name must be ASCII"):
        headers.patch_set({"X-\u00e9": "a"})


def test_headers_patch_set_with_non_latin1_value_raises_value_error() -> None:
    headers = Headers.from_items([("X-Test", "a")])

    with pytest.raises(ValueError, match="Latin-1 range"):
        headers.patch_set({"X-Test": "\u2192"})


_HEADER_NAMES = st.text(alphabet="abAB-", min_size=1, max_size=4)
_HEADER_VALUES = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789 ", max_size=8
)
_HEADER_ITEMS = st.lists(st.tuples(_HEADER_NAMES, _HEADER_VALUES), max_size=6)
_PatchHistory = tuple[list[HeaderItem], list[dict[str, str]]]


@st.composite
def _patch_values(draw: st.DrawFn) -> dict[str, str]:
    items = draw(
        st.lists(
            st.tuples(_HEADER_NAMES, _HEADER_VALUES),
            unique_by=lambda item: item[0].lower(),
            max_size=4,
        )
    )
    return dict(items)


@st.composite
def _patch_histories(draw: st.DrawFn) -> _PatchHistory:
    return draw(_HEADER_ITEMS), draw(st.lists(_patch_values(), max_size=5))


def _model_patch_set(
    model: list[HeaderItem],
    values: dict[str, str],
) -> list[HeaderItem]:
    lowered = {name.lower() for name in values}
    kept = [item for item in model if item[0].lower() not in lowered]
    return kept + list(values.items())


def _apply_history(
    history: _PatchHistory,
) -> tuple[Headers, list[HeaderItem]]:
    initial, patches = history
    headers = Headers.from_items(initial)
    model = list(initial)
    for patch in patches:
        headers = headers.patch_set(patch)
        model = _model_patch_set(model, patch)
    return headers, model


def _probe_names(history: _PatchHistory, extra: str) -> set[str]:
    initial, patches = history
    names = {extra}
    names.update(name for name, _ in initial)
    for patch in patches:
        names.update(patch)
    return {
        variant
        for name in names
        for variant in (name, name.lower(), name.upper(), name.swapcase())
    }


@given(history=_patch_histories())
def test_headers_items_after_patch_history_match_naive_model(
    history: _PatchHistory,
) -> None:
    headers, model = _apply_history(history)

    assert list(headers.items()) == model


@given(history=_patch_histories(), probe=_HEADER_NAMES)
def test_headers_lookups_after_patch_history_match_naive_model(
    history: _PatchHistory,
    probe: str,
) -> None:
    headers, model = _apply_history(history)

    for name in _probe_names(history, probe):
        expected = [
            value
            for item_name, value in model
            if item_name.lower() == name.lower()
        ]
        assert headers.get_all(name) == (expected or None)
        assert headers.get_all(name, []) == expected
        assert headers.get(name) == (expected[0] if expected else None)
        assert (name in headers) == bool(expected)
        if expected:
            assert headers[name] == expected[0]
        else:
            with pytest.raises(KeyError):
                headers[name]


@given(history=_patch_histories())
def test_headers_patched_equals_flat_headers_with_same_items(
    history: _PatchHistory,
) -> None:
    patched, model = _apply_history(history)
    flat = Headers.from_items(model)

    assert patched == flat
    assert flat == patched
    assert hash(patched) == hash(flat)


@given(
    history=_patch_histories(),
    name=_HEADER_NAMES.filter(lambda name: name.swapcase() != name),
    values=st.tuples(_HEADER_VALUES, _HEADER_VALUES),
)
def test_headers_patch_set_case_colliding_names_raises_value_error(
    history: _PatchHistory,
    name: str,
    values: tuple[str, str],
) -> None:
    headers, _ = _apply_history(history)
    patch = {name: values[0], name.swapcase(): values[1]}

    with pytest.raises(ValueError, match="duplicate header name"):
        headers.patch_set(patch)


@given(first=_patch_histories(), second=_patch_histories())
def test_headers_equality_across_histories_follows_item_equality(
    first: _PatchHistory,
    second: _PatchHistory,
) -> None:
    headers_a, model_a = _apply_history(first)
    headers_b, model_b = _apply_history(second)

    assert (headers_a == headers_b) == (model_a == model_b)
    if model_a == model_b:
        assert hash(headers_a) == hash(headers_b)
