import asyncio
import time

import httpx

URL = "http://localhost:8000/chat"  # Replace with your actual chat endpoint
NUM_REQUESTS = 3

PAYLOAD = {"user_message": "What projects has Kasturi built?"}


async def send_request(client, index):
    start = time.perf_counter()

    try:
        response = await client.post(URL, json=PAYLOAD)
        elapsed = time.perf_counter() - start

        print(
            f"Request {index}: "
            f"status={response.status_code}, "
            f"time={elapsed:.2f}s, "
            f"body={response.text[:300]!r}"
        )

    except Exception as exc:
        elapsed = time.perf_counter() - start
        print(
            f"Request {index}: error={type(exc).__name__}: {exc}, time={elapsed:.2f}s"
        )


async def main():
    async with httpx.AsyncClient(timeout=120.0) as client:
        start = time.perf_counter()

        await asyncio.gather(
            *(send_request(client, i) for i in range(1, 11))
        )

        print(f"Total wall time: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
