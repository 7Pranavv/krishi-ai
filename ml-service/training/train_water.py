# ============================================================
# WATER MANAGEMENT MODEL - TRAINING SCRIPT
# ============================================================
# What this file does:
#   1. Generates a synthetic dataset based on FAO-56 crop
#      water requirement standards (real agronomic values)
#   2. Trains a Random Forest Regressor to predict:
#      - Daily water requirement (mm/day)
#      - Irrigation interval (days)
#   3. Saves the model as water_model.pkl
#
# FAO-56 Reference: Food and Agriculture Organization of the
# United Nations - Crop Evapotranspiration Guidelines
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

np.random.seed(42)

# ── STEP 1: Define FAO-56 crop coefficients (Kc) ─────────────
# Kc values determine water need at each growth stage
# Source: FAO Irrigation and Drainage Paper No. 56
# Format: crop -> (Kc_initial, Kc_mid, Kc_late, total_days)

FAO_CROPS = {
    "rice":        (1.05, 1.20, 0.90, 120),
    "wheat":       (0.30, 1.15, 0.40, 120),
    "maize":       (0.30, 1.20, 0.60, 125),
    "sugarcane":   (0.40, 1.25, 0.75, 365),
    "cotton":      (0.35, 1.20, 0.60, 180),
    "soybean":     (0.40, 1.15, 0.50, 120),
    "potato":      (0.45, 1.10, 0.75, 105),
    "tomato":      (0.60, 1.15, 0.80, 125),
    "onion":       (0.50, 1.05, 0.75, 150),
    "chickpea":    (0.40, 1.00, 0.35, 100),
    "groundnut":   (0.45, 1.05, 0.60, 130),
    "sunflower":   (0.35, 1.10, 0.35, 125),
    "banana":      (0.50, 1.10, 1.00, 300),
    "mango":       (0.40, 0.90, 0.70, 365),
    "mustard":     (0.35, 1.10, 0.40, 110),
}

SOIL_TYPES = {
    "sandy":       {"water_holding": 0.10, "infiltration": "high",   "interval_factor": 0.7},
    "loamy":       {"water_holding": 0.20, "infiltration": "medium", "interval_factor": 1.0},
    "clay":        {"water_holding": 0.30, "infiltration": "low",    "interval_factor": 1.4},
    "sandy_loam":  {"water_holding": 0.15, "infiltration": "medium", "interval_factor": 0.85},
    "clay_loam":   {"water_holding": 0.25, "infiltration": "low",    "interval_factor": 1.2},
}

GROWTH_STAGES = ["initial", "development", "mid", "late"]

# ── STEP 2: Generate synthetic dataset ───────────────────────
# For each combination of crop, soil, stage, temperature,
# humidity, and rainfall — compute water requirement using
# Penman-Monteith simplified formula (ETo * Kc)

records = []

for _ in range(6000):
    crop       = np.random.choice(list(FAO_CROPS.keys()))
    soil       = np.random.choice(list(SOIL_TYPES.keys()))
    stage_idx  = np.random.randint(0, 4)
    stage      = GROWTH_STAGES[stage_idx]
    temp       = np.random.uniform(15, 42)       # °C
    humidity   = np.random.uniform(20, 95)       # %
    wind_speed = np.random.uniform(0.5, 5.0)     # m/s
    sunshine   = np.random.uniform(4, 12)        # hours/day
    rainfall   = np.random.uniform(0, 15)        # mm/day

    # Simplified ETo (Reference Evapotranspiration) mm/day
    # Based on Hargreaves equation approximation
    eto = 0.0023 * (temp + 17.8) * sunshine * 0.408 * (1 - humidity / 200)
    eto = max(1.5, min(eto, 12.0))

    # Kc based on growth stage
    kc_vals = FAO_CROPS[crop]
    if stage == "initial":
        kc = kc_vals[0]
    elif stage == "development":
        kc = (kc_vals[0] + kc_vals[1]) / 2
    elif stage == "mid":
        kc = kc_vals[1]
    else:
        kc = kc_vals[2]

    # Crop water requirement (ETc) mm/day
    etc = eto * kc

    # Effective rainfall (only ~70% is usable by crops)
    eff_rain = rainfall * 0.70

    # Net irrigation requirement
    net_irr = max(0, etc - eff_rain)

    # Irrigation interval based on soil water holding capacity
    soil_data = SOIL_TYPES[soil]
    root_depth = 0.4 if stage in ["initial", "development"] else 0.6  # meters
    taw = soil_data["water_holding"] * root_depth * 1000  # mm
    interval = (taw * 0.5) / (etc + 0.001)  # days before 50% depletion
    interval = interval * soil_data["interval_factor"]
    interval = max(1, min(round(interval), 14))

    # Add noise to simulate real-world variation
    net_irr += np.random.normal(0, 0.3)
    net_irr = max(0, round(net_irr, 2))

    records.append({
        "crop":        crop,
        "soil_type":   soil,
        "growth_stage": stage,
        "temperature": round(temp, 1),
        "humidity":    round(humidity, 1),
        "wind_speed":  round(wind_speed, 1),
        "sunshine_hours": round(sunshine, 1),
        "rainfall":    round(rainfall, 1),
        "water_req_mm_day": net_irr,
        "irrigation_interval_days": interval,
    })

df = pd.DataFrame(records)
print(f"✅ Dataset generated: {df.shape[0]} rows")
print(f"   Average water requirement: {df['water_req_mm_day'].mean():.2f} mm/day")
print(f"   Average irrigation interval: {df['irrigation_interval_days'].mean():.1f} days\n")

# ── STEP 3: Encode categorical features ──────────────────────
le_crop  = LabelEncoder()
le_soil  = LabelEncoder()
le_stage = LabelEncoder()

df["crop_enc"]  = le_crop.fit_transform(df["crop"])
df["soil_enc"]  = le_soil.fit_transform(df["soil_type"])
df["stage_enc"] = le_stage.fit_transform(df["growth_stage"])

features = ["crop_enc", "soil_enc", "stage_enc",
            "temperature", "humidity", "wind_speed",
            "sunshine_hours", "rainfall"]

X = df[features]
y_water    = df["water_req_mm_day"]
y_interval = df["irrigation_interval_days"]

# ── STEP 4: Train models ──────────────────────────────────────
X_train, X_test, yw_train, yw_test, yi_train, yi_test = train_test_split(
    X, y_water, y_interval, test_size=0.2, random_state=42
)

print("⏳ Training water requirement model...")
model_water = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model_water.fit(X_train, yw_train)

print("⏳ Training irrigation interval model...")
model_interval = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model_interval.fit(X_train, yi_train)

# ── STEP 5: Evaluate ─────────────────────────────────────────
yw_pred = model_water.predict(X_test)
yi_pred = model_interval.predict(X_test)

print(f"\n🎯 Water Requirement Model:")
print(f"   MAE:  {mean_absolute_error(yw_test, yw_pred):.3f} mm/day")
print(f"   R²:   {r2_score(yw_test, yw_pred):.4f}")

print(f"\n🎯 Irrigation Interval Model:")
print(f"   MAE:  {mean_absolute_error(yi_test, yi_pred):.2f} days")
print(f"   R²:   {r2_score(yi_test, yi_pred):.4f}")

# ── STEP 6: Save everything ───────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(model_water,    os.path.join(BASE_DIR, "water_model.pkl"))
joblib.dump(model_interval, os.path.join(BASE_DIR, "interval_model.pkl"))
joblib.dump(le_crop,        os.path.join(BASE_DIR, "le_crop.pkl"))
joblib.dump(le_soil,        os.path.join(BASE_DIR, "le_soil.pkl"))
joblib.dump(le_stage,       os.path.join(BASE_DIR, "le_stage.pkl"))

print("\n✅ Saved: water_model.pkl")
print("✅ Saved: interval_model.pkl")
print("✅ Saved: le_crop.pkl, le_soil.pkl, le_stage.pkl")
print("\n🚀 Run the app:")
print("   streamlit run app.py")
