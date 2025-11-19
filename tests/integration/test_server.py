import httpx
import pytest
import pytest_asyncio

from localstub.server import AsyncHTTPTestServer


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
        from localstub.server import StubResponse

        return StubResponse.json({"count": request_count})

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
