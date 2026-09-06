from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from localstub.http.connection import (
    response_allows_reuse,
    should_close_connection,
)
from localstub.http.headers import Headers
from localstub.http.request import HTTPRequest, RecordedHTTPRequest
from localstub.http.response import ParsedResponse

_CONNECTION_TOKENS = ("close", "keep-alive", "upgrade", "custom")
_CONNECTION_HEADER_NAMES = (b"Connection", b"connection", b"CONNECTION")
_TOKEN_WHITESPACE = ("", " ", "\t", "  ")


def _request(
    http_version: str = "1.1",
    headers: dict[str, str] | None = None,
) -> RecordedHTTPRequest:
    request = HTTPRequest(
        method="GET",
        target="/",
        headers=Headers.from_items((headers or {}).items()),
    )
    return RecordedHTTPRequest(
        request=request,
        as_received=request,
        wire_raw_bytes=b"",
        http_version=http_version,
    )


def test_should_close_connection_http10_response_without_keep_alive():
    assert should_close_connection(
        _request(),
        response_headers={},
        response_version="1.0",
    )


def test_should_close_connection_http10_response_with_keep_alive_persists():
    assert not should_close_connection(
        _request(),
        response_headers={"Connection": "keep-alive"},
        response_version="1.0",
    )


def test_should_close_connection_http10_response_honors_request_close():
    assert should_close_connection(
        _request(headers={"Connection": "close"}),
        response_headers={"Connection": "keep-alive"},
        response_version="1.0",
    )


def test_should_close_connection_http11_response_version_persists():
    assert not should_close_connection(
        _request(),
        response_headers={},
        response_version="1.1",
    )


def test_should_close_connection_unknown_response_version_persists():
    assert not should_close_connection(
        _request(),
        response_headers={},
    )


def _parsed_response(
    *,
    http_version: str | None = "1.1",
    status_code: int | None = 200,
    headers: list[tuple[bytes, bytes]] | None = None,
    is_complete: bool = True,
    is_eof_delimited: bool = False,
) -> ParsedResponse:
    return ParsedResponse(
        status_code=status_code,
        http_version=http_version,
        headers=headers if headers is not None else [],
        is_complete=is_complete,
        is_eof_delimited=is_eof_delimited,
    )


@st.composite
def _connection_header_cases(
    draw: st.DrawFn,
) -> tuple[list[tuple[bytes, bytes]], bool]:
    token_groups = draw(
        st.lists(
            st.lists(
                st.sampled_from(_CONNECTION_TOKENS),
                max_size=4,
            ),
            max_size=3,
        )
    )
    headers: list[tuple[bytes, bytes]] = []
    if draw(st.booleans()):
        headers.append((b"X-Unrelated", b"close"))

    for tokens in token_groups:
        wire_tokens: list[str] = []
        for token in tokens:
            variant = draw(
                st.sampled_from([token, token.upper(), token.title()])
            )
            prefix = draw(st.sampled_from(_TOKEN_WHITESPACE))
            suffix = draw(st.sampled_from(_TOKEN_WHITESPACE))
            wire_tokens.append(f"{prefix}{variant}{suffix}")
        name = draw(st.sampled_from(_CONNECTION_HEADER_NAMES))
        headers.append((name, ",".join(wire_tokens).encode("ascii")))

    has_close = any(
        token == "close" for tokens in token_groups for token in tokens
    )
    return headers, has_close


@given(
    is_complete=st.booleans(),
    is_eof_delimited=st.booleans(),
    http_version=st.sampled_from([None, "1.0", "1.1", "2.0"]),
    status_code=st.one_of(
        st.just(101),
        st.integers(min_value=100, max_value=599),
    ),
    header_case=_connection_header_cases(),
)
def test_response_allows_reuse_matches_policy_for_arbitrary_metadata(
    is_complete: bool,
    is_eof_delimited: bool,
    http_version: str | None,
    status_code: int,
    header_case: tuple[list[tuple[bytes, bytes]], bool],
) -> None:
    headers, has_close = header_case
    parsed = _parsed_response(
        http_version=http_version,
        status_code=status_code,
        headers=headers,
        is_complete=is_complete,
        is_eof_delimited=is_eof_delimited,
    )
    expected = (
        is_complete
        and not is_eof_delimited
        and http_version == "1.1"
        and status_code != 101
        and not has_close
    )

    assert response_allows_reuse(parsed) == expected


def test_response_allows_reuse_complete_http11_response():
    assert response_allows_reuse(_parsed_response())


def test_response_allows_reuse_rejects_incomplete_response():
    assert not response_allows_reuse(_parsed_response(is_complete=False))


def test_response_allows_reuse_rejects_eof_delimited_response():
    assert not response_allows_reuse(_parsed_response(is_eof_delimited=True))


def test_response_allows_reuse_rejects_http10_even_with_keep_alive():
    assert not response_allows_reuse(
        _parsed_response(
            http_version="1.0",
            headers=[(b"Connection", b"keep-alive")],
        )
    )


def test_response_allows_reuse_rejects_connection_close():
    assert not response_allows_reuse(
        _parsed_response(headers=[(b"Connection", b"close")])
    )


def test_response_allows_reuse_rejects_close_among_other_tokens():
    assert not response_allows_reuse(
        _parsed_response(headers=[(b"connection", b"keep-alive, close")])
    )


def test_response_allows_reuse_allows_keep_alive_token():
    assert response_allows_reuse(
        _parsed_response(headers=[(b"Connection", b"keep-alive")])
    )


def test_response_allows_reuse_rejects_protocol_switch():
    assert not response_allows_reuse(_parsed_response(status_code=101))
