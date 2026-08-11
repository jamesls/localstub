from __future__ import annotations

import pytest

from localstub.http.headers import Headers


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
