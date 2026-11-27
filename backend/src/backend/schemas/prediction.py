from pydantic import BaseModel


class PredictionResponse(BaseModel):
    filename: str
    width: int
    height: int
    flooded_fraction: float
    mask_png_base64: str