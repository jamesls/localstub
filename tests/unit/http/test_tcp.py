from __future__ import annotations

from localstub.http.tcp import pack_linger_option


def test_pack_linger_option_uses_shorts_on_windows() -> None:
    assert len(pack_linger_option(platform="win32")) == 4


def test_pack_linger_option_uses_ints_elsewhere() -> None:
    assert len(pack_linger_option(platform="linux")) == 8
    assert len(pack_linger_option(platform="darwin")) == 8
