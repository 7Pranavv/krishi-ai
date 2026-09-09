"""Water management - two RandomForestRegressors (daily requirement, interval).

Feature order is fixed by train_water.py:
[crop_enc, soil_enc, stage_enc, temperature, humidity, wind_speed,
 sunshine_hours, rainfall]
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from .. import knowledge
from ..deps import encode, require_bundle
from ..schemas import OptionsResponse, WaterRequest, WaterResponse

router = APIRouter(prefix="/water", tags=["water"])

_OPTION_SOURCES = {
    "crop": ("le_crop", None),
    "soil_type": ("le_soil", knowledge.SOIL_LABELS),
    "growth_stage": ("le_stage", knowledge.GROWTH_STAGE_LABELS),
}


@router.get("/options", response_model=dict[str, OptionsResponse])
def options() -> dict[str, OptionsResponse]:
    bundle = require_bundle("water")
    result = {}
    for field, (alias, label_map) in _OPTION_SOURCES.items():
        values = sorted(str(c) for c in bundle[alias].classes_)
        result[field] = OptionsResponse(
            values=values,
            labels={v: (label_map or {}).get(v, v.replace("_", " ").title()) for v in values},
        )
    return result


def _risk(water_req: float) -> tuple[str, str]:
    if water_req < 2:
        return "Low", "Soil moisture is adequate. Monitor and irrigate only if rainfall drops."
    if water_req < 5:
        return "Moderate", "Moderate water stress risk. Irrigate on schedule to maintain yield."
    return "High", "High water demand. Ensure consistent irrigation to prevent crop stress."


@router.post("/predict", response_model=WaterResponse)
def predict(payload: WaterRequest) -> WaterResponse:
    bundle = require_bundle("water")

    features = np.array([[
        encode(bundle["le_crop"], payload.crop, "crop"),
        encode(bundle["le_soil"], payload.soil_type, "soil_type"),
        encode(bundle["le_stage"], payload.growth_stage, "growth_stage"),
        payload.temperature, payload.humidity, payload.wind_speed,
        payload.sunshine_hours, payload.rainfall,
    ]], dtype=float)

    water_req = max(0.0, float(bundle["model_water"].predict(features)[0]))
    interval = max(1, round(float(bundle["model_interval"].predict(features)[0])))

    hectares = payload.area * knowledge.AREA_UNITS[payload.area_unit]
    litres_per_day = water_req * hectares * 10_000  # 1 mm over 1 ha = 10,000 L

    risk, note = _risk(water_req)

    return WaterResponse(
        water_req_mm_day=round(water_req, 2),
        irrigation_interval_days=interval,
        area_hectares=round(hectares, 4),
        litres_per_day=round(litres_per_day, 1),
        litres_per_week=round(litres_per_day * 7, 1),
        stress_risk=risk,
        risk_note=note,
        tip=knowledge.WATER_CROP_TIPS.get(payload.crop),
        schedule=[(day % interval) == 0 for day in range(7)],
    )
