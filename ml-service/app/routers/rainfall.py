"""Monthly rainfall prediction - GradientBoostingRegressor over IMD normals.

Feature order is fixed by train_rainfall.py:
[state_enc, month, season_enc, temperature, humidity, prev_month_rain,
 avg_normal_rain]
"""
from __future__ import annotations

import numpy as np
from fastapi import APIRouter, HTTPException, status

from .. import knowledge
from ..deps import encode, require_bundle
from ..schemas import OptionsResponse, RainfallRequest, RainfallResponse

router = APIRouter(prefix="/rainfall", tags=["rainfall"])


@router.get("/options", response_model=dict[str, OptionsResponse])
def options() -> dict[str, OptionsResponse]:
    bundle = require_bundle("rainfall")
    states = sorted(str(c) for c in bundle["le_state"].classes_)
    return {
        "state": OptionsResponse(values=states, labels={s: s for s in states}),
        "month": OptionsResponse(
            values=[str(i) for i in range(1, 13)],
            labels={str(i): m for i, m in enumerate(knowledge.MONTHS, start=1)},
        ),
    }


@router.post("/predict", response_model=RainfallResponse)
def predict(payload: RainfallRequest) -> RainfallResponse:
    bundle = require_bundle("rainfall")
    normals: dict[str, list[float]] = bundle["normals"]

    state_enc = encode(bundle["le_state"], payload.state, "state")
    if payload.state not in normals:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": f"No IMD normals recorded for {payload.state!r}."},
        )

    season = knowledge.MONTH_SEASON[payload.month]
    season_enc = encode(bundle["le_season"], season, "season")
    monthly = normals[payload.state]
    normal_rain = float(monthly[payload.month - 1])

    # Default the previous month to that month's IMD normal, as the original
    # tool's number_input did.
    prev_rain = payload.prev_month_rain
    if prev_rain is None:
        prev_rain = float(monthly[(payload.month - 2) % 12])

    features = np.array([[state_enc, payload.month, season_enc,
                          payload.temperature, payload.humidity,
                          prev_rain, normal_rain]], dtype=float)

    predicted = max(0.0, float(bundle["model"].predict(features)[0]))

    # IMD departure classification: +/-20% around the long-period average.
    departure = ((predicted - normal_rain) / (normal_rain + 1)) * 100
    if departure < -20:
        category = "Deficient"
    elif departure > 20:
        category = "Excess"
    else:
        category = "Normal"

    return RainfallResponse(
        state=payload.state,
        month=knowledge.MONTHS[payload.month - 1],
        season=season,
        predicted_mm=round(predicted, 1),
        normal_mm=round(normal_rain, 1),
        departure_pct=round(departure, 1),
        category=category,
        advice=knowledge.RAINFALL_ADVICE[category][season],
        monthly_normals=[float(v) for v in monthly],
    )
