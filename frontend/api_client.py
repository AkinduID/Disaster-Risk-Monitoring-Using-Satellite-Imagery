import os

import httpx

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


def predict(filename: str, data: bytes, content_type: str) -> dict:
    response = httpx.post(
        f"{BACKEND_URL}/predict",
        files={"file": (filename, data, content_type)},
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()