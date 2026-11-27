import base64
import io

import numpy as np
from PIL import Image


def predict_mask(image: Image.Image) -> np.ndarray:
    """Placeholder 'model': dark pixels count as water (SAR water is dark).

    Returns a boolean array of shape (height, width).
    Replace the body with real inference later; the signature stays the same.
    """
    gray = np.asarray(image.convert("L"))
    return gray < 80


def mask_to_png_base64(mask: np.ndarray) -> str:
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")