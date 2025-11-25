import asyncio
import time

import httpx
import pytest
import pytest_asyncio

from localstub.server import (
    AsyncHTTPTestServer,
    HTTPResponse,
    ThrottledTransmission,
)


@pytest_asyncio.fixture
async def server():
    """Fixture that provides an auto-started AsyncHTTPTestServer."""
    async with AsyncHTTPTestServer() as srv:
        yield srv


@pytest_asyncio.fixture
async def client():
    """Fixture that provides an httpx AsyncClient."""
    async with httpx.AsyncClient() as c:
        yield c


@pytest.mark.asyncio
async def test_server_receives_request_and_returns_json_response(
    server, client
):
    server.set_json_response({"message": "hello world"})

    response = await client.get(server.url)

    assert response.status_code == 200
    assert response.json() == {"message": "hello world"}
    assert response.headers["Content-Type"] == "application/json"

    assert server.last_request is not None
    assert server.last_request.method == "GET"
    assert server.last_request.path == "/"
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_server_returns_custom_headers(server, client):
    server.set_json_response(
        {"status": "ok"}, headers={"x-custom-header": "custom-value"}
    )

    response = await client.get(server.url)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-custom-header"] == "custom-value"
    assert response.headers["Content-Type"] == "application/json"


@pytest.mark.asyncio
async def test_server_records_multiple_requests(server, client):
    server.set_json_response({"response": "ok"})

    await client.get(f"{server.url}first")
    await client.post(f"{server.url}second", json={"data": "test"})
    await client.put(f"{server.url}third", content=b"raw data")

    assert len(server.requests) == 3

    assert server.requests[0].method == "GET"
    assert server.requests[0].path == "/first"

    assert server.requests[1].method == "POST"
    assert server.requests[1].path == "/second"
    assert server.requests[1].json_body == {"data": "test"}

    assert server.requests[2].method == "PUT"
    assert server.requests[2].path == "/third"
    assert server.requests[2].body == "raw data"

    assert server.last_request is server.requests[2]
    assert server.last_request.method == "PUT"


@pytest.mark.asyncio
async def test_server_records_custom_request_headers(server, client):
    server.set_json_response({"status": "ok"})

    await client.get(
        server.url,
        headers={
            "x-custom-header": "custom-value",
            "x-another-header": "another-value",
        },
    )

    assert server.last_request is not None
    assert server.last_request.headers is not None
    assert server.last_request.headers["x-custom-header"] == "custom-value"
    assert server.last_request.headers["x-another-header"] == "another-value"


@pytest.mark.asyncio
async def test_server_tracks_requests_by_client(server):
    server.set_json_response({"status": "ok"})

    async with (
        httpx.AsyncClient() as client1,
        httpx.AsyncClient() as client2,
    ):
        await client1.get(f"{server.url}client1_A")
        await client2.get(f"{server.url}client2_A")
        await client1.post(f"{server.url}client1_B", json={"id": 1})
        await client2.post(f"{server.url}client2_B", json={"id": 2})

    assert len(server.requests) == 4

    assert server.requests[0].path == "/client1_A"
    assert server.requests[1].path == "/client2_A"
    assert server.requests[2].path == "/client1_B"
    assert server.requests[3].path == "/client2_B"

    client1_addr = server.requests[0].client
    client2_addr = server.requests[1].client

    assert client1_addr is not None
    assert client2_addr is not None
    assert client1_addr != client2_addr

    client1_requests = [
        req for req in server.requests if req.client == client1_addr
    ]
    client2_requests = [
        req for req in server.requests if req.client == client2_addr
    ]

    assert len(client1_requests) == 2
    assert client1_requests[0].path == "/client1_A"
    assert client1_requests[0].method == "GET"
    assert client1_requests[1].path == "/client1_B"
    assert client1_requests[1].method == "POST"
    assert client1_requests[1].json_body == {"id": 1}

    assert len(client2_requests) == 2
    assert client2_requests[0].path == "/client2_A"
    assert client2_requests[0].method == "GET"
    assert client2_requests[1].path == "/client2_B"
    assert client2_requests[1].method == "POST"
    assert client2_requests[1].json_body == {"id": 2}


@pytest.mark.asyncio
async def test_clear_requests_resets_state(server, client):
    """Test that clear_requests() clears all recorded request state."""
    server.set_json_response({"status": "ok"})

    # Make several requests
    await client.get(f"{server.url}first")
    await client.post(f"{server.url}second", json={"data": "test"})

    # Verify requests were recorded
    assert len(server.requests) == 2
    assert server.last_request is not None
    assert server.last_request.path == "/second"

    # Clear state
    server.clear_requests()

    # Verify state is cleared
    assert len(server.requests) == 0
    assert server.last_request is None


@pytest.mark.asyncio
async def test_clear_requests_allows_multiple_cycles(server, client):
    """Test that clear_requests() can be called multiple times."""
    server.set_json_response({"cycle": 1})

    # Cycle 1
    await client.get(f"{server.url}cycle1")
    assert len(server.requests) == 1
    assert server.requests[0].path == "/cycle1"

    server.clear_requests()

    # Cycle 2
    await client.get(f"{server.url}cycle2")
    assert len(server.requests) == 1
    assert server.requests[0].path == "/cycle2"

    server.clear_requests()

    # Cycle 3
    await client.get(f"{server.url}cycle3")
    assert len(server.requests) == 1
    assert server.requests[0].path == "/cycle3"


@pytest.mark.asyncio
async def test_clear_requests_preserves_default_response(server, client):
    """Test that clear_requests() preserves the default response config."""
    server.set_json_response({"message": "configured"}, status=201)

    # Make a request and verify response
    response1 = await client.get(server.url)
    assert response1.status_code == 201
    assert response1.json() == {"message": "configured"}

    # Clear and make another request
    server.clear_requests()

    response2 = await client.get(server.url)
    assert response2.status_code == 201
    assert response2.json() == {"message": "configured"}
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_clear_requests_preserves_handler(server, client):
    """Test that clear_requests() preserves the custom handler."""

    request_count = 0

    def custom_handler(request):
        nonlocal request_count
        request_count += 1
        return HTTPResponse.json({"count": request_count})

    server.handler = custom_handler

    # First request
    response1 = await client.get(server.url)
    assert response1.json() == {"count": 1}

    # Clear and make second request
    server.clear_requests()

    response2 = await client.get(server.url)
    # Handler should still be active
    assert response2.json() == {"count": 2}
    # But requests list should only have 1 request
    assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_session_scoped_server_pattern():
    """Test session-scoped server reuse pattern across multiple tests."""
    # Simulates a session-scoped fixture
    async with AsyncHTTPTestServer() as server:
        async with httpx.AsyncClient() as client:
            # Test 1: Check initial state
            server.set_json_response({"test": 1})
            response = await client.get(server.url)
            assert response.json() == {"test": 1}
            assert len(server.requests) == 1

            # Clear state between tests
            server.clear_requests()

            # Test 2: Fresh state after clear
            server.set_json_response({"test": 2})
            response = await client.get(server.url)
            assert response.json() == {"test": 2}
            assert len(server.requests) == 1
            assert server.requests[0].path == "/"

            # Clear state between tests
            server.clear_requests()

            # Test 3: Multiple requests in one test
            server.set_json_response({"test": 3})
            await client.get(f"{server.url}a")
            await client.get(f"{server.url}b")
            assert len(server.requests) == 2
            assert server.requests[0].path == "/a"
            assert server.requests[1].path == "/b"


@pytest.mark.asyncio
async def test_connection_bytes_received_single_request(server, client):
    """Test that connection-level bytes are captured for a single request."""
    server.set_json_response({"status": "ok"})

    await client.get(f"{server.url}test")

    assert server.last_request is not None
    client_addr = server.last_request.client
    assert client_addr is not None

    # Get connection-level bytes
    conn_bytes = server.get_connection_bytes_received(client_addr)
    assert conn_bytes is not None
    assert len(conn_bytes) > 0
    # Should contain the request line
    assert b"GET /test HTTP/1.1" in conn_bytes
    # Header snippets we expect to see in the request.
    assert b"Host: 127" in conn_bytes
    assert b"User-Agent: python-httpx" in conn_bytes
    assert b"Host:" in conn_bytes


@pytest.mark.asyncio
async def test_connection_bytes_sent_single_request(server, client):
    """Test that connection-level sent bytes are captured."""
    server.set_json_response({"status": "ok"})

    await client.get(server.url)

    assert server.last_request is not None
    client_addr = server.last_request.client
    assert client_addr is not None

    # Get connection-level sent bytes
    sent_bytes = server.get_connection_bytes_sent(client_addr)
    assert sent_bytes is not None
    assert len(sent_bytes) > 0

    # Should contain HTTP status line
    assert b"HTTP/1.1 200 OK" in sent_bytes
    # Should contain the JSON response body
    assert (
        b'{"status": "ok"}' in sent_bytes or b'{"status":"ok"}' in sent_bytes
    )


@pytest.mark.asyncio
async def test_connection_bytes_multiple_requests_same_connection(
    server, client
):
    """Test that connection bytes accumulate across multiple requests."""
    server.set_json_response({"response": "ok"})

    # Make multiple requests on the same connection
    await client.get(f"{server.url}first")
    await client.post(f"{server.url}second", json={"data": "test"})
    await client.put(f"{server.url}third", content=b"raw")

    assert len(server.requests) == 3
    client_addr = server.requests[0].client
    assert client_addr is not None

    # All requests should be from the same client address
    assert all(req.client == client_addr for req in server.requests)

    # Get connection-level bytes
    conn_bytes_received = server.get_connection_bytes_received(client_addr)
    conn_bytes_sent = server.get_connection_bytes_sent(client_addr)

    assert conn_bytes_received is not None
    assert conn_bytes_sent is not None

    # Should contain all three requests
    assert b"GET /first HTTP/1.1" in conn_bytes_received
    assert b"POST /second HTTP/1.1" in conn_bytes_received
    assert b"PUT /third HTTP/1.1" in conn_bytes_received

    # Should contain request bodies
    assert (
        b'"data": "test"' in conn_bytes_received
        or b'"data":"test"' in conn_bytes_received
    )
    assert b"raw" in conn_bytes_received

    # Should contain multiple responses (3 status lines)
    assert conn_bytes_sent.count(b"HTTP/1.1 200 OK") == 3


@pytest.mark.asyncio
async def test_connection_bytes_separate_clients():
    """Test that connection bytes are tracked separately per client."""
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"status": "ok"})

        async with (
            httpx.AsyncClient() as client1,
            httpx.AsyncClient() as client2,
        ):
            # Each client makes requests
            await client1.get(f"{server.url}client1")
            await client2.get(f"{server.url}client2")

        assert len(server.requests) == 2

        client1_addr = server.requests[0].client
        client2_addr = server.requests[1].client

        assert client1_addr is not None
        assert client2_addr is not None
        assert client1_addr != client2_addr

        # Get connection bytes for each client
        client1_bytes = server.get_connection_bytes_received(client1_addr)
        client2_bytes = server.get_connection_bytes_received(client2_addr)

        assert client1_bytes is not None
        assert client2_bytes is not None

        # Each client should only have their own request
        assert b"/client1" in client1_bytes
        assert b"/client1" not in client2_bytes

        assert b"/client2" in client2_bytes
        assert b"/client2" not in client1_bytes


@pytest.mark.asyncio
async def test_connection_bytes_cleared_with_clear_requests(server, client):
    """Test that clear_requests() also clears connection bytes."""
    server.set_json_response({"status": "ok"})

    await client.get(server.url)

    assert server.last_request is not None
    client_addr = server.last_request.client
    assert client_addr is not None

    # Verify connection bytes exist
    conn_bytes = server.get_connection_bytes_received(client_addr)
    assert conn_bytes is not None
    assert len(conn_bytes) > 0

    # Clear requests
    server.clear_requests()

    # Connection bytes should be cleared
    conn_bytes_after = server.get_connection_bytes_received(client_addr)
    assert conn_bytes_after is None


@pytest.mark.asyncio
async def test_connection_bytes_match_per_request_bytes(server, client):
    """Test connection bytes include the same data as per-request bytes."""
    server.set_json_response({"status": "ok"})

    await client.post(f"{server.url}test", json={"key": "value"})

    assert server.last_request is not None
    client_addr = server.last_request.client
    assert client_addr is not None

    # Get both per-request and connection-level bytes
    request_bytes = server.last_request.wire_raw_bytes
    conn_bytes = server.get_connection_bytes_received(client_addr)

    assert request_bytes is not None
    assert conn_bytes is not None

    # For a single request, connection bytes should equal request bytes
    assert conn_bytes == request_bytes


@pytest.mark.asyncio
async def test_router_matches_method_and_path(server, client):
    """Router dispatches to method+path routes before fallback/default."""

    async def get_handler(request):
        return HTTPResponse.text("GET-OK")

    def post_handler(request):
        return HTTPResponse.json({"method": "POST"})

    server.add_route("GET", "/r1", get_handler)
    server.add_route("POST", "/r1", post_handler)

    r_get = await client.get(f"{server.url}r1")
    assert r_get.status_code == 200
    assert r_get.text == "GET-OK"

    r_post = await client.post(f"{server.url}r1")
    assert r_post.status_code == 200
    assert r_post.json() == {"method": "POST"}


@pytest.mark.asyncio
async def test_router_falls_back_to_handler_when_no_route(server, client):
    """If no route matches, server.handler handles the request."""

    def fallback_handler(request):
        return HTTPResponse.text("fallback")

    def only_handler(request):
        return HTTPResponse.text("only")

    server.handler = fallback_handler
    server.add_route("GET", "/only", only_handler)

    r_only = await client.get(f"{server.url}only")
    assert r_only.status_code == 200
    assert r_only.text == "only"

    r_other = await client.get(f"{server.url}other")
    assert r_other.status_code == 200
    assert r_other.text == "fallback"


@pytest.mark.asyncio
async def test_router_defaults_when_no_handler_and_no_route(server, client):
    """If neither route nor handler present, default response is used."""
    # Ensure no handler is set explicitly
    server.handler = None

    # Register a different path so '/unmatched' is not routed
    server.add_route("GET", "/special", lambda r: HTTPResponse.text("special"))

    r_unmatched = await client.get(f"{server.url}unmatched")
    assert r_unmatched.status_code == 200
    # Default response is JSON {}
    assert r_unmatched.json() == {}


@pytest.mark.asyncio
async def test_server_receives_chunked_request_body(server, client):
    server.set_json_response({"status": "received"})

    async def chunked_body():
        yield b"hello"
        yield b" "
        yield b"world"

    response = await client.post(server.url, content=chunked_body())

    assert response.status_code == 200
    assert server.last_request is not None
    assert server.last_request.body == "hello world"
    assert server.last_request.headers is not None
    assert (
        "chunked"
        in server.last_request.headers.get("Transfer-Encoding", "").lower()
    )


@pytest.mark.asyncio
async def test_server_handles_multiple_chunks_varying_sizes(server, client):
    server.set_json_response({"chunks": "received"})

    async def multi_chunk_body():
        yield b"a"  # 1 byte
        yield b"bb"  # 2 bytes
        yield b"ccc"  # 3 bytes
        yield b"dddd"  # 4 bytes

    response = await client.post(server.url, content=multi_chunk_body())

    assert response.status_code == 200
    assert server.last_request is not None
    assert server.last_request.body == "abbcccdddd"

    # Verify raw wire bytes contain chunk size prefixes
    wire_bytes = server.last_request.wire_raw_bytes
    assert wire_bytes is not None
    # Should contain hex chunk sizes: 1, 2, 3, 4, and final 0
    assert b"1\r\n" in wire_bytes
    assert b"2\r\n" in wire_bytes
    assert b"3\r\n" in wire_bytes
    assert b"4\r\n" in wire_bytes
    assert b"0\r\n" in wire_bytes


@pytest.mark.asyncio
async def test_server_receives_large_chunked_body(server, client):
    server.set_json_response({"size": "received"})

    async def large_body():
        # Simulate streaming 100 chunks of 1024 bytes each
        for i in range(100):
            chunk_prefix = f"chunk{i}:".encode()
            padding = b"x" * (1024 - len(chunk_prefix))
            yield chunk_prefix + padding

    response = await client.post(server.url, content=large_body())

    assert response.status_code == 200
    assert server.last_request is not None
    assert len(server.last_request.body) == 100 * 1024
    assert "chunk0:" in server.last_request.body
    assert "chunk99:" in server.last_request.body


@pytest.mark.asyncio
async def test_server_receives_json_via_chunked_encoding(server, client):
    server.set_json_response({"status": "parsed"})

    json_payload = b'{"key": "value", "nested": {"data": 123}}'

    async def chunked_json():
        # Split JSON into 3 chunks
        chunk_size = len(json_payload) // 3
        for i in range(0, len(json_payload), chunk_size):
            yield json_payload[i : i + chunk_size]

    response = await client.post(
        server.url,
        content=chunked_json(),
        headers={"Content-Type": "application/json"},
    )

    assert response.status_code == 200
    assert server.last_request is not None
    assert server.last_request.json_body == {
        "key": "value",
        "nested": {"data": 123},
    }


@pytest.mark.asyncio
async def test_connection_bytes_with_chunked_encoding(server, client):
    server.set_json_response({"status": "ok"})

    async def chunked_body():
        yield b"first"
        yield b"second"

    await client.post(server.url, content=chunked_body())

    assert server.last_request is not None
    client_addr = server.last_request.client
    assert client_addr is not None

    # Get connection-level bytes
    conn_bytes = server.get_connection_bytes_received(client_addr)
    assert conn_bytes is not None
    assert len(conn_bytes) > 0

    # Should contain chunked transfer encoding header
    assert b"Transfer-Encoding: chunked" in conn_bytes
    # Should contain chunk size prefixes in hex
    assert b"5\r\n" in conn_bytes  # "first" = 5 bytes
    assert b"6\r\n" in conn_bytes  # "second" = 6 bytes
    # Should contain the actual chunk data
    assert b"first" in conn_bytes
    assert b"second" in conn_bytes
    # Should contain final chunk marker
    assert b"0\r\n" in conn_bytes


@pytest.mark.asyncio
async def test_response_sequence_returns_responses_in_order(server, client):
    """Test that response sequence returns configured responses in order."""
    server.set_response_sequence([
        HTTPResponse(status=500),
        HTTPResponse(status=502),
        HTTPResponse.json({"success": True}),
    ])

    # First request gets 500
    response1 = await client.get(server.url)
    assert response1.status_code == 500

    # Second request gets 502
    response2 = await client.get(server.url)
    assert response2.status_code == 502

    # Third request gets 200 with JSON
    response3 = await client.get(server.url)
    assert response3.status_code == 200
    assert response3.json() == {"success": True}

    # All requests should be recorded
    assert len(server.requests) == 3


@pytest.mark.asyncio
async def test_response_sequence_exhaustion_falls_back_to_default(
    server, client
):
    """Test sequence falls back to handler/default after exhaustion."""
    # Set a handler as fallback
    server.handler = lambda req: HTTPResponse.json({"fallback": True})

    # Set a sequence of 2 responses
    server.set_response_sequence([
        HTTPResponse(status=500),
        HTTPResponse(status=500),
    ])

    # First two requests consume the sequence
    response1 = await client.get(server.url)
    assert response1.status_code == 500

    response2 = await client.get(server.url)
    assert response2.status_code == 500

    # Third request should fall back to handler
    response3 = await client.get(server.url)
    assert response3.status_code == 200
    assert response3.json() == {"fallback": True}


@pytest.mark.asyncio
async def test_response_sequence_clears_default_response(server, client):
    """Test that set_response_sequence clears previous default response."""
    # First set a default response
    server.set_json_response({"default": "value"})

    # Verify it works
    response1 = await client.get(server.url)
    assert response1.json() == {"default": "value"}

    server.clear_requests()

    # Now set a response sequence - should clear the default
    server.set_response_sequence([HTTPResponse(status=503)])

    # Should get 503, not the old default
    response2 = await client.get(server.url)
    assert response2.status_code == 503

    # After sequence exhaustion, should get empty default
    response3 = await client.get(server.url)
    assert response3.status_code == 200
    assert response3.json() == {}


@pytest.mark.asyncio
async def test_set_json_response_clears_sequence(server, client):
    """Test that set_json_response clears any response sequence."""
    # First set a sequence
    server.set_response_sequence([
        HTTPResponse(status=500),
        HTTPResponse(status=500),
    ])

    # Now set a JSON response - should clear the sequence
    server.set_json_response({"message": "override"})

    # Should get the JSON response, not sequence
    response = await client.get(server.url)
    assert response.status_code == 200
    assert response.json() == {"message": "override"}

    # Multiple requests should all get the same response
    response2 = await client.get(server.url)
    assert response2.json() == {"message": "override"}


@pytest.mark.asyncio
async def test_set_text_response_clears_sequence(server, client):
    """Test that set_text_response clears any response sequence."""
    server.set_response_sequence([HTTPResponse(status=404)])
    server.set_text_response("text override")

    response = await client.get(server.url)
    assert response.status_code == 200
    assert response.text == "text override"


@pytest.mark.asyncio
async def test_set_raw_response_clears_sequence(server, client):
    """Test that set_raw_response clears any response sequence."""
    server.set_response_sequence([HTTPResponse(status=500)])
    server.set_raw_response(b"raw bytes")

    response = await client.get(server.url)
    assert response.status_code == 200
    assert response.content == b"raw bytes"


@pytest.mark.asyncio
async def test_clear_requests_resets_sequence_index(server, client):
    """Test that clear_requests allows sequence reuse."""
    server.set_response_sequence([
        HTTPResponse(status=500),
        HTTPResponse.json({"attempt": 2}),
    ])

    # First cycle
    response1 = await client.get(server.url)
    assert response1.status_code == 500

    response2 = await client.get(server.url)
    assert response2.json() == {"attempt": 2}

    # Clear requests - should reset sequence index
    server.clear_requests()

    # Second cycle - sequence should restart
    response3 = await client.get(server.url)
    assert response3.status_code == 500

    response4 = await client.get(server.url)
    assert response4.json() == {"attempt": 2}


@pytest.mark.asyncio
async def test_response_sequence_for_retry_testing():
    """Test the motivating use case: testing client retry behavior."""
    async with AsyncHTTPTestServer() as server:
        # Configure server to fail twice, then succeed
        server.set_response_sequence([
            HTTPResponse(status=500),  # First attempt fails
            HTTPResponse(status=500),  # First retry fails
            HTTPResponse.json({"success": True}),  # Second retry succeeds
        ])

        # Simulate a client with retry logic
        async with httpx.AsyncClient() as client:
            attempts = 0
            max_attempts = 3

            for attempt in range(max_attempts):
                attempts += 1
                response = await client.get(server.url)

                if response.status_code == 200:
                    # Success!
                    assert response.json() == {"success": True}
                    break

                # Retry on 500
                assert response.status_code == 500

            # Should have made exactly 3 attempts
            assert attempts == 3
            assert len(server.requests) == 3

            # Verify the sequence worked correctly
            assert server.requests[0].method == "GET"
            assert server.requests[1].method == "GET"
            assert server.requests[2].method == "GET"


@pytest.mark.asyncio
async def test_response_sequence_respects_router_priority(server, client):
    """Test that response sequence has higher priority than routes."""
    # Set up a route
    server.add_route(
        "GET", "/special", lambda req: HTTPResponse.text("route-handler")
    )

    # Set a response sequence
    server.set_response_sequence([HTTPResponse.json({"sequence": True})])

    # Sequence should take precedence
    response = await client.get(f"{server.url}special")
    assert response.status_code == 200
    assert response.json() == {"sequence": True}


@pytest.mark.asyncio
async def test_response_sequence_works_across_multiple_clients():
    """Test that response sequence is global across different clients."""
    async with AsyncHTTPTestServer() as server:
        server.set_response_sequence([
            HTTPResponse.json({"client": 1}),
            HTTPResponse.json({"client": 2}),
            HTTPResponse.json({"client": 3}),
        ])

        # Three different clients each make one request
        async with httpx.AsyncClient() as client1:
            response1 = await client1.get(server.url)
            assert response1.json() == {"client": 1}

        async with httpx.AsyncClient() as client2:
            response2 = await client2.get(server.url)
            assert response2.json() == {"client": 2}

        async with httpx.AsyncClient() as client3:
            response3 = await client3.get(server.url)
            assert response3.json() == {"client": 3}

        # All three requests recorded
        assert len(server.requests) == 3


@pytest.mark.asyncio
async def test_response_sequence_with_different_request_methods(
    server, client
):
    """Test that sequence works regardless of HTTP method."""
    server.set_response_sequence([
        HTTPResponse.json({"method": "first"}),
        HTTPResponse.json({"method": "second"}),
        HTTPResponse.json({"method": "third"}),
    ])

    # Different methods all consume the sequence
    response1 = await client.get(server.url)
    assert response1.json() == {"method": "first"}

    response2 = await client.post(server.url, json={"test": "data"})
    assert response2.json() == {"method": "second"}

    response3 = await client.put(server.url, content=b"test")
    assert response3.json() == {"method": "third"}

    # Verify all requests recorded with correct methods
    assert server.requests[0].method == "GET"
    assert server.requests[1].method == "POST"
    assert server.requests[2].method == "PUT"


@pytest.mark.asyncio
async def test_empty_response_sequence_uses_fallback(server, client):
    """Test that an empty sequence immediately falls back."""
    server.set_json_response({"default": True})
    server.set_response_sequence([])  # Empty sequence

    response = await client.get(server.url)
    # Should use the default (which was cleared by set_response_sequence)
    assert response.status_code == 200
    assert response.json() == {}


async def send_raw_request(host, port, data):
    """Send raw bytes to server and read response."""
    reader, writer = await asyncio.open_connection(host, port)
    try:
        writer.write(data)
        await writer.drain()
        # Give server time to process
        await asyncio.sleep(0.1)
        response = await asyncio.wait_for(reader.read(4096), timeout=1.0)
        return response
    except asyncio.TimeoutError:
        return b""
    finally:
        writer.close()
        await writer.wait_closed()


@pytest.mark.asyncio
async def test_server_handles_empty_request_line(server):
    """Test server handles empty request line gracefully."""
    # Send just EOF without any request line
    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.close()
    await writer.wait_closed()

    # Server should not crash, no request should be recorded
    await asyncio.sleep(0.1)
    assert len(server.requests) == 0


@pytest.mark.asyncio
async def test_server_handles_http09_simple_request(server):
    """Test server handles HTTP/0.9 simple request format (no version)."""
    # HTTP/0.9 simple request format: "GET /path\r\n"
    # httptools correctly parses this as HTTP/0.9
    await send_raw_request(
        server.host,
        server.port,
        b"GET /path\r\n\r\n",
    )

    # Server should handle HTTP/0.9 requests correctly
    await asyncio.sleep(0.1)
    assert len(server.requests) == 1
    assert server.requests[0].method == "GET"
    assert server.requests[0].path == "/path"
    assert server.requests[0].http_version == "0.9"


@pytest.mark.asyncio
async def test_server_handles_eof_while_reading_headers(server):
    """Test server handles EOF while reading headers."""
    # Send request line but close before sending complete headers
    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(b"GET / HTTP/1.1\r\n")
    writer.write(b"Host: localhost\r\n")
    await writer.drain()
    writer.close()
    await writer.wait_closed()

    await asyncio.sleep(0.1)
    # Request should not be recorded since headers weren't complete
    assert len(server.requests) == 0


@pytest.mark.asyncio
async def test_server_handles_invalid_content_length(server):
    """Test server handles non-numeric Content-Length.

    httptools strictly validates HTTP headers per spec, so invalid
    Content-Length values cause a parse failure. The connection is
    closed without recording the request.
    """
    response = await send_raw_request(
        server.host,
        server.port,
        b"POST / HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Content-Length: not-a-number\r\n"
        b"\r\n",
    )

    # httptools rejects invalid Content-Length (strict HTTP compliance)
    # Connection is closed without sending a response
    await asyncio.sleep(0.1)
    assert len(server.requests) == 0
    assert response == b""


@pytest.mark.asyncio
async def test_server_handles_invalid_chunk_size(server):
    """Test server handles invalid chunk size in chunked encoding."""
    await send_raw_request(
        server.host,
        server.port,
        b"POST / HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"INVALID\r\n"
        b"test\r\n",
    )

    # Server should handle this gracefully
    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_server_handles_eof_in_chunked_body(server):
    """Test server handles EOF while reading chunked body."""
    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(
        b"POST / HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\n"
    )
    await writer.drain()
    # Close before sending the chunk data
    writer.close()
    await writer.wait_closed()

    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_server_handles_eof_in_chunk_trailers(server):
    """Test server handles EOF while reading chunk trailers."""
    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(
        b"POST / HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nhello\r\n"
        b"0\r\n"
    )
    await writer.drain()
    # Close before sending final CRLF
    writer.close()
    await writer.wait_closed()

    await asyncio.sleep(0.1)


@pytest.mark.asyncio
async def test_server_handles_invalid_status_code():
    """Test server handles invalid status codes gracefully."""
    async with AsyncHTTPTestServer() as server:
        # Set response with invalid status code
        server.set_json_response({"test": "value"}, status=999)

        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()

        response = await reader.read(4096)
        writer.close()
        await writer.wait_closed()

        # Server should return the response with "UNKNOWN" reason
        assert b"HTTP/1.1 999 UNKNOWN" in response


# ---------------------------------------------------------------------------
# next_request() timeout tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_next_request_without_timeout(server):
    """Test next_request() waits indefinitely without timeout."""

    async def make_request():
        await asyncio.sleep(0.1)
        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        await reader.read(4096)
        writer.close()
        await writer.wait_closed()

    # Start request in background
    request_task = asyncio.create_task(make_request())

    # Wait for request without timeout
    request = await server.next_request(timeout=None)
    assert request.method == "GET"
    assert request.path == "/"

    await request_task


@pytest.mark.asyncio
async def test_next_request_with_timeout_success(server):
    """Test next_request() with timeout that completes in time."""

    async def make_request():
        await asyncio.sleep(0.05)
        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        writer.write(b"GET /test HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        await reader.read(4096)
        writer.close()
        await writer.wait_closed()

    request_task = asyncio.create_task(make_request())

    # Wait with sufficient timeout
    request = await server.next_request(timeout=5.0)
    assert request.method == "GET"
    assert request.path == "/test"

    await request_task


@pytest.mark.asyncio
async def test_next_request_with_timeout_expires():
    """Test next_request() raises TimeoutError when timeout expires."""
    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"test": "ok"})

        # Wait for request that never arrives
        with pytest.raises(asyncio.TimeoutError):
            await server.next_request(timeout=0.1)


# ---------------------------------------------------------------------------
# Exception handling tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_handles_exception_during_request_processing():
    """Test server handles exceptions during request processing."""
    async with AsyncHTTPTestServer() as server:

        def failing_handler(request):
            raise ValueError("Handler error")

        server.handler = failing_handler

        # Make request that will trigger handler exception
        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()

        # Server should handle exception and close connection
        await asyncio.sleep(0.2)

        writer.close()
        await writer.wait_closed()

        # Request should still be recorded before handler error
        assert len(server.requests) == 1


@pytest.mark.asyncio
async def test_server_handles_writer_close_exception():
    """Test server handles exceptions when closing writer."""

    async with AsyncHTTPTestServer() as server:
        server.set_json_response({"test": "ok"})

        # Make a normal request
        reader, writer = await asyncio.open_connection(
            server.host, server.port
        )
        writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()

        # Read response
        await reader.read(4096)

        # Forcefully close from client side
        writer.close()
        await writer.wait_closed()

        # Give server time to handle cleanup
        await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Raw socket tests for EOF in various stages
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_handles_immediate_eof_in_request_line(server):
    """Test server handles immediate EOF (no data at all)."""
    reader, writer = await asyncio.open_connection(server.host, server.port)
    # Close immediately without sending anything
    writer.close()
    await writer.wait_closed()

    await asyncio.sleep(0.1)
    assert len(server.requests) == 0


@pytest.mark.asyncio
async def test_server_handles_eof_after_chunk_size(server):
    """Test server handles EOF right after reading chunk size line."""
    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(
        b"POST / HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
    )
    await writer.drain()
    # Close before sending any chunk size
    writer.close()
    await writer.wait_closed()

    await asyncio.sleep(0.1)


# ---------------------------------------------------------------------------
# Connection: close handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_closes_connection_on_connection_close_header(server):
    """Test server closes connection when Connection: close is sent."""
    reader, writer = await asyncio.open_connection(server.host, server.port)

    # Send request with Connection: close
    writer.write(
        b"GET / HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n"
    )
    await writer.drain()

    # Read response
    response = await reader.read(4096)
    assert b"HTTP/1.1 200 OK" in response
    assert b"Connection: close" in response

    # Try to send another request on the same connection
    writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
    await writer.drain()

    # Server should have closed the connection after first request
    # Second request should get no response
    await asyncio.sleep(0.1)

    writer.close()
    try:
        await writer.wait_closed()
    except ConnectionResetError:
        # Expected - server already closed the connection
        pass

    # Only first request should be recorded
    assert len(server.requests) == 1


# ---------------------------------------------------------------------------
# Additional edge case tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_server_handles_chunked_with_trailer_headers(server):
    """Test server handles chunked encoding with trailing headers."""
    reader, writer = await asyncio.open_connection(server.host, server.port)
    writer.write(
        b"POST / HTTP/1.1\r\n"
        b"Host: localhost\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\n"
        b"hello\r\n"
        b"0\r\n"
        b"X-Trailer: value\r\n"
        b"\r\n"
    )
    await writer.drain()

    response = await reader.read(4096)
    assert b"HTTP/1.1 200 OK" in response

    writer.close()
    await writer.wait_closed()

    await asyncio.sleep(0.1)
    assert len(server.requests) == 1
    assert server.last_request.body == "hello"


# ---------------------------------------------------------------------------
# Transmission strategy tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_throttled_transmission_slows_response(server, client):
    """Test ThrottledTransmission delays response body transmission."""
    response_data = b"x" * 10000  # 10KB of data

    server.set_raw_response(response_data)
    server.set_transmission_strategy(
        ThrottledTransmission(chunk_size=1000, delay=0.05)
    )

    start = time.time()
    response = await client.get(server.url)
    elapsed = time.time() - start

    # Verify response is correct
    assert response.status_code == 200
    assert response.content == response_data

    # Should have taken roughly: 10 chunks with 9 delays = 9 * 0.05 = ~0.45s
    # Allow some margin for network/processing overhead
    assert elapsed >= 0.40

    # Request should be recorded normally
    assert len(server.requests) == 1
    assert server.last_request.method == "GET"


@pytest.mark.asyncio
async def test_throttled_transmission_large_chunks():
    """Test throttled transmission with larger chunk size."""
    async with AsyncHTTPTestServer() as server:
        response_data = b"y" * 50000  # 50KB

        server.set_raw_response(response_data)
        server.set_transmission_strategy(
            ThrottledTransmission(chunk_size=10000, delay=0.02)
        )

        async with httpx.AsyncClient() as client:
            start = time.time()
            response = await client.get(server.url)
            elapsed = time.time() - start

            # 50KB / 10KB = 5 chunks, 4 delays = 4 * 0.02 = 0.08s
            assert elapsed >= 0.07
            assert response.content == response_data


@pytest.mark.asyncio
async def test_throttled_transmission_small_body(server, client):
    """Test throttled transmission with body smaller than chunk size."""
    small_data = b"small response"

    server.set_raw_response(small_data)
    server.set_transmission_strategy(
        ThrottledTransmission(chunk_size=1000, delay=0.1)
    )

    start = time.time()
    response = await client.get(server.url)
    elapsed = time.time() - start

    # Single chunk, no delay
    assert elapsed < 0.05
    assert response.content == small_data


@pytest.mark.asyncio
async def test_throttled_transmission_json_response(server, client):
    """Test throttled transmission works with JSON responses."""
    json_data = {"data": "x" * 5000}  # Large JSON

    server.set_json_response(json_data)
    server.set_transmission_strategy(
        ThrottledTransmission(chunk_size=500, delay=0.01)
    )

    start = time.time()
    response = await client.get(server.url)
    elapsed = time.time() - start

    # Should have throttled the transmission
    assert elapsed > 0.01
    assert response.json() == json_data


@pytest.mark.asyncio
async def test_throttled_transmission_with_handler(server, client):
    """Test throttled transmission works with custom handlers."""

    def handler(request):
        return HTTPResponse.raw(b"handler response" * 1000)

    server.handler = handler
    server.set_transmission_strategy(
        ThrottledTransmission(chunk_size=1000, delay=0.02)
    )

    start = time.time()
    response = await client.get(server.url)
    elapsed = time.time() - start

    expected = b"handler response" * 1000
    assert response.content == expected
    # ~16KB with 1KB chunks = 16 chunks, 15 delays = 0.3s
    assert elapsed >= 0.25


@pytest.mark.asyncio
async def test_throttled_transmission_with_multiple_requests(server, client):
    """Test throttled transmission applies to all requests."""
    server.set_raw_response(b"x" * 5000)
    server.set_transmission_strategy(
        ThrottledTransmission(chunk_size=1000, delay=0.02)
    )

    # Make multiple requests - all should be throttled
    for i in range(3):
        start = time.time()
        response = await client.get(server.url)
        elapsed = time.time() - start

        assert response.status_code == 200
        # 5 chunks, 4 delays = 0.08s
        assert elapsed >= 0.07

    assert len(server.requests) == 3


@pytest.mark.asyncio
async def test_default_transmission_is_immediate(server, client):
    """Test that default transmission (without throttling) is fast."""
    response_data = b"z" * 10000

    server.set_raw_response(response_data)
    # Don't set any transmission strategy - should use default immediate

    start = time.time()
    response = await client.get(server.url)
    elapsed = time.time() - start

    # Should be very fast (no artificial delays)
    assert elapsed < 0.1
    assert response.content == response_data
