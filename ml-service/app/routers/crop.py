"""Crop recommendation - RandomForestClassifier over the Kaggle crop dataset.

Feature order is fixed by train_crop.py: [N, P, K, temperature, humidity, ph, rainfall]

The prediction is also annotated with the season for the farmer's region and,
where SHAP is available, which inputs drove the choice.
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, Query

from .. import explain, knowledge, season
from ..config import LOW_CONFIDENCE_THRESHOLD
from ..deps import require_bundle
from ..schemas import (CropCandidate, CropRequest, CropResponse, OptionsResponse,
                       SeasonResponse)

router = APIRouter(prefix="/crop", tags=["crop"])

FEATURE_NAMES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


@router.get("/options", response_model=OptionsResponse)
def options() -> OptionsResponse:
    bundle = require_bundle("crop")
    crops = sorted(str(c) for c in bundle["le"].classes_)
    return OptionsResponse(
        values=crops,
        labels={c: knowledge.CROP_INFO.get(c, {}).get("emoji", "") for c in crops},
    )


@router.get("/season", response_model=SeasonResponse)
def current_season(
    month: int | None = Query(None, ge=1, le=12),
    state: str | None = Query(None, max_length=64),
) -> SeasonResponse:
    """Which season it is for a state, and what grows in it."""
    return SeasonResponse(**season.detect(month, state))


@router.post("/predict", response_model=CropResponse)
def predict(
    payload: CropRequest,
    state: str | None = Query(None, max_length=64,
                              description="Used only to pick the regional season calendar"),
    month: int | None = Query(None, ge=1, le=12),
) -> CropResponse:
    bundle = require_bundle("crop")
    model, le = bundle["model"], bundle["le"]

    features = np.array([[payload.N, payload.P, payload.K, payload.temperature,
                          payload.humidity, payload.ph, payload.rainfall]], dtype=float)

    probabilities = model.predict_proba(features)[0]
    best = int(np.argmax(probabilities))
    crop = str(le.inverse_transform([best])[0])
    confidence = float(probabilities[best])

    def candidate(index: int) -> CropCandidate:
        name = str(le.inverse_transform([index])[0])
        return CropCandidate(
            crop=name,
            probability=round(float(probabilities[index]), 4),
            emoji=knowledge.CROP_INFO.get(name, {}).get("emoji"),
        )

    top = [candidate(int(i)) for i in np.argsort(probabilities)[::-1][:3]]

    current = season.detect(month, state)
    info = knowledge.CROP_INFO.get(crop, {})

    return CropResponse(
        crop=crop,
        confidence=round(confidence, 4),
        low_confidence=confidence < LOW_CONFIDENCE_THRESHOLD,
        season=info.get("season"),
        tip=info.get("tip"),
        emoji=info.get("emoji"),
        top_predictions=top,
        current_season=SeasonResponse(**current),
        season_fit=season.suitability(crop, current["season"]),
        drivers=explain.explain_crop(model, features, FEATURE_NAMES, best),
    )
