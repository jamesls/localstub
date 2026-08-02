from __future__ import annotations

import ssl
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Self

import trustme
from cryptography import x509
from cryptography.hazmat.primitives.serialization import (
    BestAvailableEncryption,
    pkcs12,
)

DEFAULT_CONTEXT_CACHE_SIZE = 128


def _write_private_key(path: Path, key_bytes: bytes) -> None:
    # NamedTemporaryFile creates the file with mode 0o600. The umask can only
    # remove permissions, and replacing the destination preserves that mode.
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}-",
        delete=False,
    ) as key_file:
        temporary_path = Path(key_file.name)
        key_file.write(key_bytes)

    try:
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


class TLSProxyCA:
    """CA used for TLS proxy, issues per-host server contexts on demand.

    Issued contexts are cached per hostname so repeated connections to the
    same host reuse a single certificate and ``SSLContext``. The cache is
    bounded by ``context_cache_size`` and evicts the least recently issued
    host first.
    """

    _ca: trustme.CA
    _ca_pem_path: Path
    _ca_pkcs12_path: Path

    def __init__(
        self,
        *,
        ca: trustme.CA | None = None,
        pem_path: Path | None = None,
        pkcs12_path: Path | None = None,
        context_cache_size: int = DEFAULT_CONTEXT_CACHE_SIZE,
    ) -> None:
        if context_cache_size < 0:
            raise ValueError(
                f"context_cache_size must not be negative: "
                f"{context_cache_size}"
            )
        self._context_cache_size = context_cache_size
        self._context_cache: OrderedDict[str, ssl.SSLContext] = OrderedDict()
        if ca is not None:
            self._ca = ca
            if pem_path is None:
                with tempfile.NamedTemporaryFile(
                    prefix="localstub-ca-", suffix=".pem", delete=False
                ) as f:
                    f.write(self._ca.cert_pem.bytes())
                    self._ca_pem_path = Path(f.name)
            else:
                self._ca_pem_path = pem_path
            self._ca_pkcs12_path = (
                pkcs12_path
                if pkcs12_path is not None
                else self._ca_pem_path.with_suffix(".p12")
            )
            if not self._ca_pkcs12_path.exists():
                p12_bytes = self._convert_to_pkcs12_truststore(
                    self._load_pem_certificate()
                )
                self._ca_pkcs12_path.write_bytes(p12_bytes)
            return

        self._ca = trustme.CA()
        with tempfile.NamedTemporaryFile(
            prefix="localstub-ca-", suffix=".pem", delete=False
        ) as f:
            f.write(self._ca.cert_pem.bytes())
            self._ca_pem_path = Path(f.name)
        with tempfile.NamedTemporaryFile(
            prefix="localstub-ca-", suffix=".p12", delete=False
        ) as f:
            p12_bytes = self._convert_to_pkcs12_truststore(
                self._load_pem_certificate()
            )
            self._ca_pkcs12_path = Path(f.name)
            self._ca_pkcs12_path.write_bytes(p12_bytes)

    def issue_context(self, host: str) -> ssl.SSLContext:
        cached = self._context_cache.get(host)
        if cached is not None:
            self._context_cache.move_to_end(host)
            return cached
        context = self._build_context(host)
        if self._context_cache_size > 0:
            self._context_cache[host] = context
            if len(self._context_cache) > self._context_cache_size:
                self._context_cache.popitem(last=False)
        return context

    def _build_context(self, host: str) -> ssl.SSLContext:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_cert = self._ca.issue_server_cert(host)
        server_cert.configure_cert(context)
        context.set_alpn_protocols(["http/1.1"])
        context.options |= ssl.OP_NO_COMPRESSION
        return context

    def ca_pem_path(self) -> Path:
        return self._ca_pem_path

    def ca_pkcs12_path(self) -> Path:
        return self._ca_pkcs12_path

    def _load_pem_certificate(self) -> x509.Certificate:
        data = self._ca_pem_path.read_bytes()
        # Assume there's only one PEM cert, this is what trustme does
        # so we'll always just take the first cert.
        chunks = data.split(b"-----BEGIN CERTIFICATE-----")
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            pem_block = b"-----BEGIN CERTIFICATE-----" + chunk
            cert = x509.load_pem_x509_certificate(pem_block)
            return cert
        raise RuntimeError(f"No PEM certificate found in: {self.ca_pem_path}")

    def _convert_to_pkcs12_truststore(
        self,
        pem_cert: x509.Certificate,
        *,
        password: str = 'changeit',
        alias: str = 'localstub',
    ) -> bytes:
        p12_bytes = pkcs12.serialize_java_truststore(
            [pkcs12.PKCS12Certificate(pem_cert, alias.encode('utf-8'))],
            BestAvailableEncryption(password.encode('utf-8')),
        )
        return p12_bytes

    @classmethod
    def from_directory(cls, ca_dir: Path) -> Self:
        """Load or create a CA from a directory.

        If the directory contains a .pem and .key file, loads the existing CA.
        If the directory contains neither, generates a new CA and saves to
        ca.pem, ca.key, ca.p12.

        Raises ValueError if multiple .pem or .key files are found, or if
        only one of the .pem/.key pair exists (generating a new CA could
        overwrite the surviving file).
        """
        ca_dir.mkdir(parents=True, exist_ok=True)

        pem_files = list(ca_dir.glob("*.pem"))
        key_files = list(ca_dir.glob("*.key"))

        if len(pem_files) > 1:
            raise ValueError(f"Multiple .pem files in {ca_dir}: {pem_files}")
        if len(key_files) > 1:
            raise ValueError(f"Multiple .key files in {ca_dir}: {key_files}")

        if pem_files and key_files:
            ca = trustme.CA.from_pem(
                cert_bytes=pem_files[0].read_bytes(),
                private_key_bytes=key_files[0].read_bytes(),
            )
            return cls(ca=ca, pem_path=pem_files[0])

        if pem_files or key_files:
            found = (pem_files or key_files)[0]
            missing = ".key" if pem_files else ".pem"
            raise ValueError(
                f"Refusing to generate a new CA: {ca_dir} contains "
                f"{found.name} but no {missing} file. Restore the missing "
                f"file or remove {found.name} to generate a new CA."
            )

        cert_path = ca_dir / "ca.pem"
        key_path = ca_dir / "ca.key"
        pkcs12_path = ca_dir / "ca.p12"

        ca = trustme.CA()
        cert_path.write_bytes(ca.cert_pem.bytes())
        _write_private_key(key_path, ca.private_key_pem.bytes())

        return cls(ca=ca, pem_path=cert_path, pkcs12_path=pkcs12_path)
