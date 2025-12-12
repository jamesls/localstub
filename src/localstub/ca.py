import tempfile
from pathlib import Path
import ssl

from cryptography import x509
from cryptography.hazmat.primitives.serialization import (
    pkcs12,
    BestAvailableEncryption,
)
import trustme


class TLSProxyCA:
    """CA used for TLS proxy, issues per-host server contexts on demand."""

    def __init__(self) -> None:
        self._ca = trustme.CA()
        with tempfile.NamedTemporaryFile(
            prefix="localstub-ca-", suffix=".pem", delete=False
        ) as f:
            f.write(self._ca.cert_pem.bytes())
            self._ca_pem_path = Path(f.name)
        with tempfile.NamedTemporaryFile(
            prefix="localstub-ca-", suffix=".p12", delete=False
        ) as f:
            pkcs12_bytes = self._convert_to_pkcs12_truststore(
                self._load_pem_certificate()
            )
            self._ca_pkcs12_path = Path(f.name)
            self._ca_pkcs12_path.write_bytes(pkcs12_bytes)

    def issue_context(self, host: str) -> ssl.SSLContext:
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
