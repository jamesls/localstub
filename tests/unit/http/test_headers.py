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
