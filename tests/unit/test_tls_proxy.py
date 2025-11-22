import os
import tempfile
from pathlib import Path

import pytest

from localstub.tls_proxy import _TrustMeCA


def test_trustme_ca_closes_temp_fd(monkeypatch, tmp_path):
    captured: dict[str, int | str] = {}
    real_mkstemp = tempfile.mkstemp

    def tracking_mkstemp(*, prefix: str, suffix: str):
        fd, path = real_mkstemp(prefix=prefix, suffix=suffix, dir=tmp_path)
        captured["fd"] = fd
        captured["path"] = path
        return fd, path

    monkeypatch.setattr(tempfile, "mkstemp", tracking_mkstemp)

    ca = _TrustMeCA()

    fd = captured["fd"]
    path = Path(captured["path"])

    with pytest.raises(OSError):
        os.close(fd)

    assert path.exists()
    assert ca.ca_pem_path() == path
