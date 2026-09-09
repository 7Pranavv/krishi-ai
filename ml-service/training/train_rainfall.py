# ============================================================
# RAINFALL PREDICTION MODEL - TRAINING SCRIPT
# ============================================================
# What this file does:
#   1. Generates a dataset based on real IMD (India
#      Meteorological Department) historical rainfall averages
#      for 20 Indian states across all 12 months
#   2. Trains a Gradient Boosting Regressor to predict
#      monthly rainfall (mm)
#   3. Saves the model as rainfall_model.pkl
#
# Data source basis:
#   IMD historical normals (1951-2000)
#   https://imdpune.gov.in/
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

np.random.seed(42)

# ── STEP 1: Real IMD historical monthly rainfall (mm) ────────
# Format: state -> [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]
# Values based on IMD district-wise normal rainfall data

IMD_RAINFALL = {
    "Madhya Pradesh":    [14,  14,  11,   8,  11, 143, 330, 310, 196,  44,  10,   8],
    "Maharashtra":       [2,   1,   3,   7,  18, 116, 280, 250, 175,  72,  22,   5],
    "Uttar Pradesh":     [22,  18,  10,   5,  12, 100, 280, 290, 190,  50,  12,  10],
    "Punjab":            [40,  35,  25,  15,  20,  55, 200, 210, 100,  20,   8,  25],
    "Haryana":           [30,  25,  18,  10,  15,  50, 180, 190,  90,  18,   6,  20],
    "Rajasthan":         [8,   6,   4,   3,   8,  40, 130, 140,  70,  15,   4,   5],
    "Gujarat":           [2,   1,   1,   1,   5,  90, 280, 250, 130,  30,   8,   2],
    "Karnataka":         [5,   8,  12,  35,  90, 110, 120, 130, 160, 180,  60,  15],
    "Tamil Nadu":        [35,  25,  15,  20,  35,  40,  80,  90, 110, 200, 310, 140],
    "Kerala":            [20,  25,  40,  90, 200, 600, 650, 430, 270, 270, 170,  50],
    "Andhra Pradesh":    [10,  10,  10,  20,  50,  80, 130, 140, 160, 130,  60,  20],
    "West Bengal":       [15,  25,  35,  55, 130, 270, 330, 320, 250, 150,  40,  10],
    "Bihar":             [18,  20,  12,   8,  25, 150, 310, 300, 210,  80,  15,  10],
    "Odisha":            [20,  25,  25,  30,  60, 200, 340, 330, 250, 130,  50,  15],
    "Assam":             [30,  50,  90, 150, 250, 320, 380, 340, 250, 150,  50,  15],
    "Himachal Pradesh":  [80,  75,  65,  40,  55,  90, 200, 210, 130,  50,  25,  60],
    "Uttarakhand":       [70,  65,  55,  35,  60, 150, 350, 380, 220,  60,  20,  50],
    "Jharkhand":         [20,  25,  20,  20,  50, 200, 340, 320, 230, 100,  25,  10],
    "Chhattisgarh":      [15,  15,  12,  10,  25, 180, 360, 340, 230,  70,  15,   8],
    "Telangana":         [8,   8,  10,  18,  45,  90, 150, 160, 170, 120,  50,  15],
}

MONTHS = ["January","February","March","April","May","June",
          "July","August","September","October","November","December"]

SEASONS = {
    1: "Winter", 2: "Winter", 3: "Pre-Kharif", 4: "Pre-Kharif",
    5: "Pre-Kharif", 6: "Kharif", 7: "Kharif", 8: "Kharif",
    9: "Kharif", 10: "Rabi", 11: "Rabi", 12: "Rabi"
}

# ── STEP 2: Generate dataset with noise ──────────────────────
records = []

for state, monthly_avg in IMD_RAINFALL.items():
    for year_sim in range(50):          # simulate 50 years per state
        prev_rain = 0
        for month_idx, avg_rain in enumerate(monthly_avg):
            month_num = month_idx + 1

            # Simulate year-to-year variability (±30%)
            variability = np.random.normal(1.0, 0.25)
            actual_rain = max(0, avg_rain * variability)

            # Temperature varies by month and state
            base_temp = 25 + 5 * np.sin((month_num - 4) * np.pi / 6)
            temp = base_temp + np.random.normal(0, 2)

            # Humidity correlates with rainfall
            humidity = 40 + (actual_rain / 10) + np.random.normal(0, 8)
            humidity = max(20, min(98, humidity))

            records.append({
                "state":        state,
                "month":        month_num,
                "season":       SEASONS[month_num],
                "temperature":  round(temp, 1),
                "humidity":     round(humidity, 1),
                "prev_month_rain": round(prev_rain, 1),
                "avg_normal_rain": avg_rain,
                "rainfall_mm":  round(actual_rain, 1),
            })
            prev_rain = actual_rain

df = pd.DataFrame(records)
print(f"✅ Dataset generated: {df.shape[0]} rows, {len(IMD_RAINFALL)} states")
print(f"   Avg rainfall: {df['rainfall_mm'].mean():.1f} mm")
print(f"   Max rainfall: {df['rainfall_mm'].max():.1f} mm\n")

# ── STEP 3: Encode categoricals ───────────────────────────────
le_state  = LabelEncoder()
le_season = LabelEncoder()

df["state_enc"]  = le_state.fit_transform(df["state"])
df["season_enc"] = le_season.fit_transform(df["season"])

features = ["state_enc", "month", "season_enc", "temperature",
            "humidity", "prev_month_rain", "avg_normal_rain"]

X = df[features]
y = df["rainfall_mm"]

# ── STEP 4: Train Gradient Boosting model ────────────────────
# GBR works better than Random Forest for sequential time-based
# patterns like monthly rainfall

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("⏳ Training Gradient Boosting model...")
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    min_samples_split=5,
    random_state=42
)
model.fit(X_train, y_train)
print("✅ Training complete!\n")

# ── STEP 5: Evaluate ─────────────────────────────────────────
y_pred = model.predict(X_test)
y_pred = np.maximum(y_pred, 0)

print(f"🎯 Model Performance:")
print(f"   MAE:  {mean_absolute_error(y_test, y_pred):.2f} mm")
print(f"   R²:   {r2_score(y_test, y_pred):.4f}")

importances = model.feature_importances_
feat_names  = ["State", "Month", "Season", "Temperature", "Humidity", "Prev Rain", "Normal Rain"]
print("\n📌 Feature Importance:")
for name, imp in sorted(zip(feat_names, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"   {name:<14} {imp*100:5.1f}%  {bar}")

# ── STEP 6: Save ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(model,    os.path.join(BASE_DIR, "rainfall_model.pkl"))
joblib.dump(le_state, os.path.join(BASE_DIR, "le_state.pkl"))
joblib.dump(le_season,os.path.join(BASE_DIR, "le_season.pkl"))

# Save the IMD normals for use in the app
import json
with open(os.path.join(BASE_DIR, "imd_normals.json"), "w") as f:
    json.dump(IMD_RAINFALL, f)

print("\n✅ Saved: rainfall_model.pkl")
print("✅ Saved: le_state.pkl, le_season.pkl")
print("✅ Saved: imd_normals.json")
print("\n🚀 Run the app:")
print("   streamlit run app.py")
