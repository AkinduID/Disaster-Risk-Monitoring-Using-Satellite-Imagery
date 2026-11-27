import io

from fastapi import APIRouter, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from backend.schemas.prediction import PredictionResponse
from backend.services.segmentation import mask_to_png_base64, predict_mask

router = APIRouter(tags=["prediction"])

ALLOWED_TYPES = {"image/png", "image/jpeg"}
MAX_BYTES = 20 * 1024 * 1024  # 20 MB


@router.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile) -> PredictionResponse:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Only PNG or JPEG images are supported")

    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 20 MB)")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Could not read the file as an image") from exc

    mask = predict_mask(image)
    return PredictionResponse(
        filename=file.filename or "upload",
        width=image.width,
        height=image.height,
        flooded_fraction=float(mask.mean()),
        mask_png_base64=mask_to_png_base64(mask),
    )