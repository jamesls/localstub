import ssl
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.serialization import pkcs12

import trustme

from localstub.ca import TLSProxyCA


def test_can_access_pem_file():
    ca = TLSProxyCA()
    contents = ca.ca_pem_path().read_text()
    assert contents.startswith("-----BEGIN CERTIFICATE-----")
    assert contents.endswith("-----END CERTIFICATE-----\n")


def test_can_issue_context_for_server():
    ca = TLSProxyCA()
    context = ca.issue_context("s3.amazonaws.com")
    assert isinstance(context, ssl.SSLContext)
    assert context.protocol == ssl.PROTOCOL_TLS_SERVER
    assert context.options & ssl.OP_NO_COMPRESSION


def test_can_convert_to_pkcs12_truststore():
    ca = TLSProxyCA()
    cert_contents = ca.ca_pkcs12_path().read_bytes()
    pkcs12_data = pkcs12.load_pkcs12(cert_contents, b"changeit")
    assert len(pkcs12_data.additional_certs) == 1
    assert pkcs12_data.additional_certs[0].friendly_name == b"localstub"


def test_constructor_honors_provided_ca_without_pem_path() -> None:
    existing_ca = trustme.CA()
    ca = TLSProxyCA(ca=existing_ca)

    assert ca.ca_pem_path().read_bytes() == existing_ca.cert_pem.bytes()


class TestFromDirectory:
    def test_generates_ca_when_directory_does_not_exist(
        self, tmp_path: Path
    ) -> None:
        ca_dir = tmp_path / "new_ca"
        ca = TLSProxyCA.from_directory(ca_dir)

        assert ca_dir.exists()
        assert (ca_dir / "ca.pem").exists()
        assert (ca_dir / "ca.key").exists()
        assert (ca_dir / "ca.p12").exists()
        assert ca.ca_pem_path() == ca_dir / "ca.pem"
        assert ca.ca_pkcs12_path() == ca_dir / "ca.p12"

    def test_generates_ca_when_directory_is_empty(
        self, tmp_path: Path
    ) -> None:
        ca_dir = tmp_path / "empty_ca"
        ca_dir.mkdir()

        TLSProxyCA.from_directory(ca_dir)

        assert (ca_dir / "ca.pem").exists()
        assert (ca_dir / "ca.key").exists()
        assert (ca_dir / "ca.p12").exists()

    def test_loads_existing_ca_from_directory(self, tmp_path: Path) -> None:
        ca_dir = tmp_path / "existing_ca"

        TLSProxyCA.from_directory(ca_dir)

        cert_path = ca_dir / "ca.pem"
        key_path = ca_dir / "ca.key"
        original_cert_bytes = cert_path.read_bytes()

        new_cert_path = ca_dir / "myroot.pem"
        new_key_path = ca_dir / "myroot.key"
        new_cert_path.write_bytes(cert_path.read_bytes())
        new_key_path.write_bytes(key_path.read_bytes())
        cert_path.unlink()
        key_path.unlink()
        (ca_dir / "ca.p12").unlink()

        loaded_ca = TLSProxyCA.from_directory(ca_dir)

        assert loaded_ca.ca_pem_path() == new_cert_path
        assert loaded_ca.ca_pkcs12_path() == ca_dir / "myroot.p12"
        assert loaded_ca.ca_pem_path().read_bytes() == original_cert_bytes

    def test_loaded_ca_can_issue_certificates(self, tmp_path: Path) -> None:
        ca_dir = tmp_path / "issuing_ca"
        TLSProxyCA.from_directory(ca_dir)

        reloaded_ca = TLSProxyCA.from_directory(ca_dir)

        context = reloaded_ca.issue_context("example.com")
        assert isinstance(context, ssl.SSLContext)

    def test_raises_error_for_multiple_pem_files(self, tmp_path: Path) -> None:
        ca_dir = tmp_path / "multi_pem"
        ca_dir.mkdir()
        (ca_dir / "first.pem").write_text("dummy")
        (ca_dir / "second.pem").write_text("dummy")
        (ca_dir / "ca.key").write_text("dummy")

        with pytest.raises(ValueError, match="Multiple .pem files"):
            TLSProxyCA.from_directory(ca_dir)

    def test_raises_error_for_multiple_key_files(self, tmp_path: Path) -> None:
        ca_dir = tmp_path / "multi_key"
        ca_dir.mkdir()
        (ca_dir / "ca.pem").write_text("dummy")
        (ca_dir / "first.key").write_text("dummy")
        (ca_dir / "second.key").write_text("dummy")

        with pytest.raises(ValueError, match="Multiple .key files"):
            TLSProxyCA.from_directory(ca_dir)
