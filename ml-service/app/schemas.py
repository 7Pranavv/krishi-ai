"""Request/response contracts for the ML service.

Numeric bounds mirror the input ranges the original Streamlit tools exposed,
which in turn come from the range each model was trained on. Values outside
them are rejected rather than silently extrapolated.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------ crop
class CropRequest(Strict):
    N: float = Field(..., ge=0, le=140, description="Available nitrogen, kg/ha")
    P: float = Field(..., ge=5, le=145, description="Available phosphorus, kg/ha")
    K: float = Field(..., ge=5, le=205, description="Available potassium, kg/ha")
    temperature: float = Field(..., ge=8, le=44, description="Average temperature, C")
    humidity: float = Field(..., ge=14, le=100, description="Relative humidity, %")
    ph: float = Field(..., ge=3.5, le=10, description="Soil pH")
    rainfall: float = Field(..., ge=20, le=300, description="Rainfall, mm")


class SeasonResponse(Strict):
    """The cropping season for a month and region, from a calendar lookup."""

    season: str
    season_label: str
    region: str
    region_label: str
    month: int
    note: str
    months_in_season: list[int]
    crops: dict[str, list[str]]


class Driver(Strict):
    """One input's contribution to a prediction, from SHAP."""

    feature: str
    label: str
    value: float
    unit: str
    impact: float
    direction: Literal["supports", "counts against"]


class CropCandidate(Strict):
    crop: str
    probability: float
    emoji: str | None = None


class CropResponse(Strict):
    crop: str
    confidence: float
    low_confidence: bool
    season: str | None = None
    tip: str | None = None
    emoji: str | None = None
    top_predictions: list[CropCandidate]
    current_season: SeasonResponse | None = None
    season_fit: Literal["highly_suitable", "suitable", "not_suitable"] | None = Field(
        None, description="How the recommended crop fits the current season")
    drivers: list[Driver] | None = Field(
        None, description="Which inputs drove this choice; null when unavailable")


# ------------------------------------------------------------ fertilizer
class FertilizerRequest(Strict):
    crop: str
    soil_type: str
    irrigation: str
    prev_crop: str
    soil_N: float = Field(..., ge=0, le=900, description="Soil test N, kg/ha")
    soil_P: float = Field(..., ge=0, le=45, description="Soil test P, kg/ha")
    soil_K: float = Field(..., ge=0, le=500, description="Soil test K, kg/ha")
    soil_pH: float = Field(..., ge=4, le=9)
    organic_matter: float = Field(..., ge=0.1, le=3.0, description="Organic matter, %")
    target_yield: float = Field(..., ge=0.1, le=100, description="Expected yield, t/ha")


class ScheduleStep(Strict):
    stage: str
    action: str


class FertilizerResponse(Strict):
    fertilizer: str
    n_dose_kg_ha: float
    p_dose_kg_ha: float
    k_dose_kg_ha: float
    urea_kg_ha: float
    dap_kg_ha: float
    mop_kg_ha: float
    soil_status: dict[str, str]
    schedule: list[ScheduleStep]
    tip: str


# ----------------------------------------------------------------- water
class WaterRequest(Strict):
    crop: str
    soil_type: str
    growth_stage: str
    temperature: float = Field(..., ge=10, le=45)
    humidity: float = Field(..., ge=10, le=100)
    wind_speed: float = Field(..., ge=0, le=8, description="m/s")
    sunshine_hours: float = Field(..., ge=2, le=14, description="hours/day")
    rainfall: float = Field(..., ge=0, le=30, description="Recent rainfall, mm/day")
    area: float = Field(1.0, gt=0, le=500, description="Field area")
    area_unit: Literal["hectare", "acre", "bigha"] = "hectare"


class WaterResponse(Strict):
    water_req_mm_day: float
    irrigation_interval_days: int
    area_hectares: float
    litres_per_day: float
    litres_per_week: float
    stress_risk: Literal["Low", "Moderate", "High"]
    risk_note: str
    tip: str | None = None
    schedule: list[bool] = Field(..., description="7 days from today; True = irrigate")


# -------------------------------------------------------------- rainfall
class RainfallRequest(Strict):
    state: str
    month: int = Field(..., ge=1, le=12)
    temperature: float = Field(..., ge=5, le=45)
    humidity: float = Field(..., ge=10, le=100)
    prev_month_rain: float | None = Field(
        None, ge=0, le=800,
        description="Last month's rainfall in mm. Defaults to the IMD normal.",
    )


class RainfallResponse(Strict):
    state: str
    month: str
    season: str
    predicted_mm: float
    normal_mm: float
    departure_pct: float
    category: Literal["Deficient", "Normal", "Excess"]
    advice: str
    monthly_normals: list[float]


# --------------------------------------------------------------- disease
class DiseaseCandidate(Strict):
    label: str
    crop: str
    condition: str
    probability: float


class DiseaseResponse(Strict):
    label: str
    crop: str
    condition: str
    healthy: bool
    confidence: float
    low_confidence: bool
    top_predictions: list[DiseaseCandidate]
    heatmap: str | None = Field(
        None, description="Grad-CAM overlay as a data-URL PNG; null when unavailable")


# ----------------------------------------------------------------- misc
class OptionsResponse(Strict):
    """Dropdown choices read straight off the trained label encoders."""

    values: list[str]
    labels: dict[str, str] = Field(default_factory=dict)

