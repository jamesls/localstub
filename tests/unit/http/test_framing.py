from localstub.http.framing import scan_chunked_body


def test_scan_chunked_body_resumes_after_complete_chunks() -> None:
    buffer = bytearray(b"3\r\none\r\n")

    first_scan = scan_chunked_body(buffer)

    assert first_scan.end is None
    assert first_scan.resume_from == len(buffer)

    buffer.extend(b"3\r\ntwo\r\n0\r\n\r\n")
    second_scan = scan_chunked_body(buffer, first_scan.resume_from)

    assert second_scan.end == len(buffer)
