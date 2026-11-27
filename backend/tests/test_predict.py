import io

from fastapi.testclient import TestClient
from PIL import Image

from backend.main import app

client = TestClient(app)


def make_png(gray_value: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (32, 32), color=gray_value).save(buffer, format="PNG")
    return buffer.getvalue()


def test_dark_image_is_all_water():
    response = client.post("/predict", files={"file": ("dark.png", make_png(0), "image/png")})
    assert response.status_code == 200
    body = response.json()
    assert body["width"] == 32
    assert body["height"] == 32
    assert body["flooded_fraction"] == 1.0


def test_bright_image_is_no_water():
    response = client.post("/predict", files={"file": ("bright.png", make_png(255), "image/png")})
    assert response.status_code == 200
    assert response.json()["flooded_fraction"] == 0.0


def test_rejects_non_image():
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 415