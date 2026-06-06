"""Prediction endpoints: POST /predict and POST /predict/compare.

Uploaded files are validated (type + size) and decoded in memory only — they
are never written to disk.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from config import ALLOWED_CONTENT_TYPES, MAX_UPLOAD_BYTES
from schemas.response import CompareResponse, PredictResponse
from services import predictor
from services.predictor import CorruptImageError

router = APIRouter()

_VALID_MODELS = {"both", "cnn", "mobilenet"}


async def _read_validated(file: UploadFile) -> bytes:
    """Validate an upload's type/size and return its bytes.

    400 for a missing file or wrong content type; 400 for oversize.
    """
    if file is None or not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG or PNG.",
        )
    raw = await file.read()
    if len(raw) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit.",
        )
    return raw


def _ensure_models_loaded(which: str) -> None:
    """500 if a requested model is not loaded (weights missing/training pending)."""
    status = predictor.registry.status
    needed = ["cnn", "mobilenet"] if which == "both" else [which]
    missing = [n for n in needed if not status.get(n, False)]
    if missing:
        raise HTTPException(
            status_code=500,
            detail=f"Model(s) not loaded: {', '.join(missing)}. Train and restart.",
        )


@router.post(
    "/predict",
    response_model=PredictResponse,
    response_model_exclude_none=True,  # omit unused model keys / null gradcam
    tags=["predict"],
)
async def predict_endpoint(
    image: UploadFile = File(None),
    model: str = Form("both"),
    gradcam: bool = Form(True),
) -> PredictResponse:
    if model not in _VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Choose one of: {sorted(_VALID_MODELS)}.",
        )

    raw = await _read_validated(image)
    _ensure_models_loaded(model)

    try:
        image_rgb = predictor.decode_image(raw)
    except CorruptImageError as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    try:
        return predictor.predict(image_rgb, which=model, want_gradcam=gradcam)
    except HTTPException:
        raise
    except Exception as exc:  # model/inference failure
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")


@router.post(
    "/predict/compare",
    response_model=CompareResponse,
    response_model_exclude_none=True,
    tags=["predict"],
)
async def compare_endpoint(
    image_a: UploadFile = File(None),
    image_b: UploadFile = File(None),
    model: str = Form("both"),
) -> CompareResponse:
    if model not in _VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Choose one of: {sorted(_VALID_MODELS)}.",
        )

    raw_a = await _read_validated(image_a)
    raw_b = await _read_validated(image_b)
    _ensure_models_loaded(model)

    try:
        img_a = predictor.decode_image(raw_a)
        img_b = predictor.decode_image(raw_b)
    except CorruptImageError as exc:
        raise HTTPException(status_code=422, detail=f"Could not decode image: {exc}")

    try:
        return predictor.compare(img_a, img_b, which=model)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")
