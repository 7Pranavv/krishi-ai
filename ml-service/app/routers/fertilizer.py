"""Fertilizer recommendation - three GradientBoostingRegressors (N/P/K dose)
plus a RandomForestClassifier that picks the commercial product.

Feature order is fixed by train_fertilizer.py:
[crop_enc, soil_enc, irr_enc, prev_enc, soil_N, soil_P, soil_K, soil_pH,
 organic_matter, target_yield]
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from .. import knowledge
from ..deps import encode, require_bundle
from ..schemas import FertilizerRequest, FertilizerResponse, OptionsResponse, ScheduleStep

router = APIRouter(prefix="/fertilizer", tags=["fertilizer"])

_OPTION_SOURCES = {
    "crop": ("le_crop", None),
    "soil_type": ("le_soil", knowledge.SOIL_LABELS),
    "irrigation": ("le_irr", knowledge.IRRIGATION_LABELS),
    "prev_crop": ("le_prev", knowledge.PREV_CROP_LABELS),
}


@router.get("/options", response_model=dict[str, OptionsResponse])
def options() -> dict[str, OptionsResponse]:
    bundle = require_bundle("fertilizer")
    result = {}
    for field, (alias, label_map) in _OPTION_SOURCES.items():
        values = sorted(str(c) for c in bundle[alias].classes_)
        result[field] = OptionsResponse(
            values=values,
            labels={v: (label_map or {}).get(v, v.replace("_", " ").title()) for v in values},
        )
    return result


@router.post("/predict", response_model=FertilizerResponse)
def predict(payload: FertilizerRequest) -> FertilizerResponse:
    bundle = require_bundle("fertilizer")

    features = np.array([[
        encode(bundle["le_crop"], payload.crop, "crop"),
        encode(bundle["le_soil"], payload.soil_type, "soil_type"),
        encode(bundle["le_irr"], payload.irrigation, "irrigation"),
        encode(bundle["le_prev"], payload.prev_crop, "prev_crop"),
        payload.soil_N, payload.soil_P, payload.soil_K, payload.soil_pH,
        payload.organic_matter, payload.target_yield,
    ]], dtype=float)

    # Regressors can dip below zero; a negative dose is meaningless.
    n_dose = max(0.0, float(bundle["model_N"].predict(features)[0]))
    p_dose = max(0.0, float(bundle["model_P"].predict(features)[0]))
    k_dose = max(0.0, float(bundle["model_K"].predict(features)[0]))
    fertilizer = str(bundle["le_fert"].inverse_transform(
        bundle["model_fert"].predict(features)
    )[0])

    content = knowledge.COMMERCIAL_CONTENT
    schedule_key = knowledge.CROP_SCHEDULE_MAP.get(payload.crop, "cereal")

    return FertilizerResponse(
        fertilizer=fertilizer,
        n_dose_kg_ha=round(n_dose, 1),
        p_dose_kg_ha=round(p_dose, 1),
        k_dose_kg_ha=round(k_dose, 1),
        urea_kg_ha=round(n_dose / content["urea"], 1),
        dap_kg_ha=round(p_dose / content["dap"], 1),
        mop_kg_ha=round(k_dose / content["mop"], 1),
        soil_status={
            "N": knowledge.soil_nutrient_status(payload.soil_N, 280, 560),
            "P": knowledge.soil_nutrient_status(payload.soil_P, 11, 22),
            "K": knowledge.soil_nutrient_status(payload.soil_K, 110, 280),
        },
        schedule=[ScheduleStep(stage=s, action=a)
                  for s, a in knowledge.FERTILIZER_SCHEDULES[schedule_key]],
        tip=knowledge.FERTILIZER_TIPS.get(payload.crop, knowledge.FERTILIZER_TIPS["default"]),
    )
