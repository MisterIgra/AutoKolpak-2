from __future__ import annotations

import os
from io import BytesIO

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageFile

from app.services.ai_service import AIService, ModelNotConfigured
from autokolpak_ai.constants import GENERATOR_VERSION

router = APIRouter(prefix="/api")
service = AIService()
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_PIXELS = 25_000_000


@router.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": service.ready, "backend": "anthropic_messages", "model": service.model, "model_version": GENERATOR_VERSION, "error": None if service.ready else "API-ключ не настроен"}


@router.get("/model-info")
def model_info() -> dict[str, object]:
    return {"ready": service.ready, "backend": "anthropic_messages", "model": service.model, "base_url": service.base_url, "capabilities": ["image_analysis", "ratio_estimation"], "status": "ready" if service.ready else "not_configured"}


@router.post("/predict")
async def predict(file: UploadFile = File(...), reference_outer_diameter_mm: float | None = None) -> JSONResponse:
    if not file.filename:
        raise HTTPException(400, "Имя файла обязательно")
    if reference_outer_diameter_mm is not None and not 330 <= reference_outer_diameter_mm <= 460:
        raise HTTPException(422, "reference_outer_diameter_mm должен быть в диапазоне 330–460 мм")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Файл превышает ограничение 15 МБ")
    try:
        with Image.open(BytesIO(data)) as source:
            if source.width * source.height > MAX_PIXELS:
                raise HTTPException(413, "Изображение превышает ограничение 25 Мп")
            source.verify()
        image = Image.open(BytesIO(data)).convert("RGB")
    except HTTPException:
        raise
    except (Image.DecompressionBombError, Image.UnidentifiedImageError, OSError) as exc:
        raise HTTPException(400, "Файл не является корректным изображением") from exc
    try:
        result = service.analyze(image)
    except ModelNotConfigured as exc:
        raise HTTPException(503, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from exc
    result["reference_outer_diameter_mm"] = reference_outer_diameter_mm
    return JSONResponse(result)