"""Plant disease detection - fine-tuned Xception classifier, 38 PlantVillage classes.

Preprocessing matches the training notebook exactly: resize to 224x224 and
rescale by 1/255 (the notebook's ImageDataGenerator used rescale=1./255, not
Xception's own preprocess_input). Changing this would silently degrade
accuracy, so it is pinned here.
"""
from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from .. import gradcam, knowledge
from ..config import LOW_CONFIDENCE_THRESHOLD, MAX_IMAGE_BYTES
from ..deps import require_bundle
from ..schemas import DiseaseCandidate, DiseaseResponse

log = logging.getLogger(__name__)
router = APIRouter(prefix="/disease", tags=["disease"])

IMG_SIZE = 224
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def _decode(raw: bytes) -> Image.Image:
    """Turn uploaded bytes into an RGB image or raise a 4xx."""
    if not raw:
        raise HTTPException(status.HTTP_400_BAD_REQUEST,
                            detail={"error": "The uploaded file is empty."})
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={"error": f"Image is larger than {MAX_IMAGE_BYTES // (1024 * 1024)} MB."},
        )
    try:
        # verify() consumes the file object, so decode from a second buffer.
        Image.open(io.BytesIO(raw)).verify()
        image = Image.open(io.BytesIO(raw))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            detail={"error": "That file is not a readable image.", "detail": str(exc)},
        ) from exc

    if image.format not in ALLOWED_FORMATS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail={"error": f"Unsupported image format {image.format}. "
                             f"Use {', '.join(sorted(ALLOWED_FORMATS))}."},
        )
    # Palette/grayscale/alpha images would otherwise produce a wrong tensor shape.
    return image.convert("RGB")


def _preprocess(image: Image.Image) -> np.ndarray:
    array = np.asarray(image.resize((IMG_SIZE, IMG_SIZE)), dtype="float32") / 255.0
    return np.expand_dims(array, axis=0)


@router.post("/predict", response_model=DiseaseResponse)
async def predict(
    file: UploadFile = File(...),
    heatmap: bool = Form(True, description="Also return a Grad-CAM overlay"),
) -> DiseaseResponse:
    bundle = require_bundle("disease")
    image = _decode(await file.read())
    tensor = _preprocess(image)

    try:
        probabilities = bundle["model"].predict(tensor, verbose=0)[0]
    except Exception as exc:  # noqa: BLE001 - inference must not 500 opaquely
        log.exception("disease inference failed")
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail={"error": "The disease model could not process this image.",
                    "detail": str(exc)},
        ) from exc

    classes: dict[int, str] = bundle["class_indices"]
    order = np.argsort(probabilities)[::-1][:3]

    def candidate(index: int) -> DiseaseCandidate:
        label = classes[index]
        crop, condition, _ = knowledge.prettify_disease_label(label)
        return DiseaseCandidate(label=label, crop=crop, condition=condition,
                                probability=round(float(probabilities[index]), 4))

    best = candidate(int(order[0]))
    _, _, healthy = knowledge.prettify_disease_label(best.label)

    overlay = (
        gradcam.heatmap_overlay(bundle["model"], tensor, image, int(order[0]))
        if heatmap else None
    )

    return DiseaseResponse(
        label=best.label,
        crop=best.crop,
        condition=best.condition,
        healthy=healthy,
        confidence=best.probability,
        low_confidence=best.probability < LOW_CONFIDENCE_THRESHOLD,
        top_predictions=[candidate(int(i)) for i in order],
        heatmap=overlay,
    )


@router.get("/classes")
def classes() -> dict[str, list[str]]:
    """The 38 labels this model can produce. Available even without weights."""
    from .. import registry

    bundle = registry.get("disease")
    labels = bundle.objects.get("class_indices")
    if not labels:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail={"error": "Class list unavailable.",
                                    "detail": bundle.error})
    return {"classes": [labels[i] for i in sorted(labels)]}
