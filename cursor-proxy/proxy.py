import json
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, Response

app = FastAPI()

UPSTREAM = "https://rmgpilot.aip.sharif.ir/v1"
API_KEY = os.environ["sharif_api"]


@app.get("/v1/models")
async def models():
    async with httpx.AsyncClient() as client:
        r = await client.get(
            f"{UPSTREAM}/models",
            headers={
                "x-litellm-api-key": API_KEY,
            },
        )

    print("\n=== MODELS ===")
    print("Status:", r.status_code)
    print(r.text[:5000])

    return Response(
        content=r.content,
        status_code=r.status_code,
        media_type="application/json",
    )


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):

    body = await request.body()

    print("\n\n" + "=" * 100)
    print("CURSOR REQUEST")
    print("=" * 100)

    try:
        data = json.loads(body)
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception:
        print("INVALID JSON:")
        print(body[:10000])

    print("=" * 100)

    client = httpx.AsyncClient(timeout=None)

    upstream_request = client.build_request(
        "POST",
        f"{UPSTREAM}/chat/completions",
        content=body,
        headers={
            "x-litellm-api-key": API_KEY,
            "Content-Type": "application/json",
        },
    )

    response = await client.send(
        upstream_request,
        stream=True,
    )

    print("\n=== UPSTREAM RESPONSE ===")
    print("Status:", response.status_code)
    print("Content-Type:", response.headers.get("content-type"))

    async def stream():
        try:
            async for chunk in response.aiter_raw():
                print("\n--- UPSTREAM CHUNK ---")
                print(repr(chunk[:1000]))
                yield chunk
        finally:
            await response.aclose()
            await client.aclose()

    return StreamingResponse(
        stream(),
        status_code=response.status_code,
        media_type=response.headers.get(
            "content-type",
            "text/event-stream",
        ),
    )
