"""ML tool endpoints.

Uniform shape for every model:

    GET  /api/ml/{tool}/options   dropdown values, read off the trained encoders
    POST /api/ml/{tool}/predict   run inference and record it in history

The backend owns auth, upload limits and history; the ML service owns the
models. Model files are never exposed to the browser.
"""
from __future__ import annotations

import logging
from urllib.parse import urlencode

from fastapi import (APIRouter, Depends, File, Form, HTTPException, Query, Request,
                     UploadFile, status)
from sqlmodel import Session

from ..config import get_settings
from ..db import get_session
from ..models import Prediction, User
from ..security import current_user
from ..services import assistant, ml_client

log = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/ml", tags=["ml"])

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


def _record(session: Session, user: User, tool: str, summary: str,
            inputs: dict, result: dict) -> None:
    """History is a convenience, never a reason to fail a good prediction."""
    try:
        session.add(Prediction(user_id=user.id, tool=tool, summary=summary[:200],
                               inputs=inputs, result=result))
        session.commit()
    except Exception:  # noqa: BLE001
        session.rollback()
        log.exception("could not save %s prediction for user %s", tool, user.id)


# --------------------------------------------------------------- options
@router.get("/{tool}/options")
async def options(tool: str) -> dict:
    if tool not in {"crop", "fertilizer", "water", "rainfall"}:
        raise HTTPException(status.HTTP_404_NOT_FOUND,
                            detail={"error": f"No options for tool {tool!r}."})
    return await ml_client.get_json(f"/{tool}/options")


@router.get("/disease/classes")
async def disease_classes() -> dict:
    return await ml_client.get_json("/disease/classes")


# ------------------------------------------------------------ tabular ML
# Bodies are validated by the ML service against the ranges each model was
# trained on; forwarding them intact keeps one source of truth for the rules.
@router.post("/crop/predict")
async def crop_predict(
    payload: dict,
    state: str | None = Query(None, max_length=64),
    month: int | None = Query(None, ge=1, le=12),
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    # state/month only select the regional season calendar; they are not model
    # inputs, which is why they travel as query parameters rather than in the
    # body the ML service validates.
    query = urlencode({k: v for k, v in (("state", state), ("month", month))
                       if v is not None})
    result = await ml_client.post_json(
        f"/crop/predict{f'?{query}' if query else ''}", payload)
    _record(session, user, "crop",
            f"Recommended {result['crop']} ({result['confidence'] * 100:.0f}% confidence)",
            payload, result)
    return result


@router.post("/fertilizer/predict")
async def fertilizer_predict(payload: dict, user: User = Depends(current_user),
                             session: Session = Depends(get_session)) -> dict:
    result = await ml_client.post_json("/fertilizer/predict", payload)
    _record(session, user, "fertilizer",
            f"{result['fertilizer']} for {payload.get('crop', 'crop')} - "
            f"N {result['n_dose_kg_ha']:.0f}, P {result['p_dose_kg_ha']:.0f}, "
            f"K {result['k_dose_kg_ha']:.0f} kg/ha",
            payload, result)
    return result


@router.post("/water/predict")
async def water_predict(payload: dict, user: User = Depends(current_user),
                        session: Session = Depends(get_session)) -> dict:
    result = await ml_client.post_json("/water/predict", payload)
    _record(session, user, "water",
            f"{payload.get('crop', 'Crop')}: {result['water_req_mm_day']} mm/day, "
            f"irrigate every {result['irrigation_interval_days']} days",
            payload, result)
    return result


@router.post("/rainfall/predict")
async def rainfall_predict(payload: dict, user: User = Depends(current_user),
                           session: Session = Depends(get_session)) -> dict:
    result = await ml_client.post_json("/rainfall/predict", payload)
    _record(session, user, "rainfall",
            f"{result['state']} {result['month']}: {result['predicted_mm']:.0f} mm "
            f"({result['category']})",
            payload, result)
    return result


# --------------------------------------------------------------- disease
@router.post("/disease/predict")
async def disease_predict(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form("English"),
    explain: bool = Form(True),
    heatmap: bool = Form(True),
    user: User = Depends(current_user),
    session: Session = Depends(get_session),
) -> dict:
    declared = (request.headers.get("content-length") or "").strip()
    if declared.isdigit() and int(declared) > settings.max_upload_bytes * 1.1:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": f"Please upload an image under "
                             f"{settings.max_upload_bytes // (1024 * 1024)} MB."},
        )

    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": "Please upload a JPG, PNG or WEBP photo.",
                    "detail": f"Received content type: {file.content_type}"},
        )

    # Read with a hard cap so a lying content-length cannot exhaust memory.
    content = await file.read(settings.max_upload_bytes + 1)
    if len(content) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": f"Please upload an image under "
                             f"{settings.max_upload_bytes // (1024 * 1024)} MB."},
        )
    if not content:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail={"error": "The uploaded file is empty."})

    result = await ml_client.post_image(
        "/disease/predict", file.filename or "upload.jpg", content, file.content_type,
        data={"heatmap": heatmap},
    )

    # Guidance is an optional extra. If the assistant is off or fails, the
    # detection is still returned - it is just not annotated.
    if explain and not result.get("healthy"):
        result["guidance"] = await assistant.explain_disease(
            result["label"], result["crop"], result["condition"], language
        )
    else:
        result["guidance"] = None

    # The Grad-CAM overlay is ~275 KB of base64. Keeping it out of the history
    # row stops the database growing by a quarter megabyte per photo; the
    # farmer still sees it in the response.
    stored = {k: v for k, v in result.items() if k != "heatmap"}
    stored["heatmap_returned"] = bool(result.get("heatmap"))

    _record(session, user, "disease",
            f"{result['crop']}: {result['condition']} "
            f"({result['confidence'] * 100:.0f}% confidence)",
            {"filename": file.filename, "language": language}, stored)
    return result
