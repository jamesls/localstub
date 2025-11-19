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
