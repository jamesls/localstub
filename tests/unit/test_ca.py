import os
import tempfile
from pathlib import Path

import pytest

from localstub.ca import TLSProxyCA
import ssl
from cryptography.hazmat.primitives.serialization import pkcs12


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
