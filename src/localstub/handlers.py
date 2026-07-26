from localstub.server import HTTPRequestHeaders, HTTPResponse, SendResponse


async def handle_expect_header(
    headers: HTTPRequestHeaders, send: SendResponse
) -> bool:
    if "100-continue" in headers.headers.get("Expect", "").lower():
        await send(HTTPResponse(status=100))
    return True
