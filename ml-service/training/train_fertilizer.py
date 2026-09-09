# ============================================================
# FERTILIZER RECOMMENDATION MODEL - TRAINING SCRIPT
# ============================================================
# Data basis:
#   - ICAR (Indian Council of Agricultural Research) crop
#     nutrient requirement tables
#   - FAO Fertilizer and Plant Nutrition Bulletin No. 19
#   - State Agricultural University recommendations (MP, UP,
#     Punjab, Maharashtra, Karnataka)
#
# What it predicts:
#   1. Recommended N dose (kg/ha)
#   2. Recommended P2O5 dose (kg/ha)
#   3. Recommended K2O dose (kg/ha)
#   4. Best fertilizer type (Urea, DAP, NPK complex, etc.)
#   5. Application stage advice
#
# Dataset: 15,000+ synthetic rows grounded in real ICAR values
# ============================================================

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

np.random.seed(42)

# ── ICAR Crop Nutrient Requirements (kg/ha) ───────────────────
# Format: crop -> (N_total, P2O5_total, K2O_total, target_yield_t_ha)
# Source: ICAR Handbook of Agriculture, 2016 edition

ICAR_NUTRIENTS = {
    # Cereals
    "rice":          (120, 60,  60,  5.0),
    "wheat":         (120, 60,  40,  4.5),
    "maize":         (150, 75,  75,  6.0),
    "sorghum":       (80,  40,  40,  3.0),
    "bajra":         (80,  40,  40,  2.5),
    "barley":        (60,  30,  20,  3.0),
    # Pulses
    "chickpea":      (20,  60,  30,  1.5),
    "lentil":        (20,  40,  20,  1.2),
    "pigeonpeas":    (20,  50,  30,  1.5),
    "mungbean":      (20,  40,  20,  1.0),
    "soybean":       (30,  60,  40,  2.0),
    # Oilseeds
    "groundnut":     (25,  50,  75,  2.5),
    "mustard":       (80,  40,  40,  2.0),
    "sunflower":     (90,  60,  60,  2.5),
    "sesame":        (40,  30,  30,  0.8),
    # Cash crops
    "sugarcane":     (250, 115, 115, 80.0),
    "cotton":        (120, 60,  60,  2.5),
    "jute":          (60,  30,  30,  3.0),
    # Vegetables
    "potato":        (180, 90,  150, 25.0),
    "tomato":        (150, 75,  75,  30.0),
    "onion":         (100, 50,  100, 20.0),
    "cabbage":       (120, 60,  60,  25.0),
    # Fruits
    "banana":        (200, 60,  200, 40.0),
    "mango":         (1000,500, 1000,10.0),  # per tree, scaled
}

SOIL_TYPES = ["sandy", "loamy", "clay", "sandy_loam", "clay_loam", "black_cotton", "red_laterite"]

# Soil nutrient availability factors — how much of applied fertilizer
# is actually available to the crop (varies by soil type)
SOIL_FACTORS = {
    "sandy":        {"N": 0.70, "P": 0.55, "K": 0.65, "pH_range": (5.5, 7.0)},
    "loamy":        {"N": 0.85, "P": 0.75, "K": 0.80, "pH_range": (6.0, 7.5)},
    "clay":         {"N": 0.80, "P": 0.65, "K": 0.85, "pH_range": (6.5, 8.0)},
    "sandy_loam":   {"N": 0.78, "P": 0.68, "K": 0.72, "pH_range": (5.8, 7.2)},
    "clay_loam":    {"N": 0.82, "P": 0.70, "K": 0.83, "pH_range": (6.2, 7.8)},
    "black_cotton": {"N": 0.75, "P": 0.60, "K": 0.90, "pH_range": (7.0, 8.5)},
    "red_laterite": {"N": 0.72, "P": 0.50, "K": 0.60, "pH_range": (5.0, 6.5)},
}

# Previous crop residual N benefit (kg/ha)
PREV_CROP_N_CREDIT = {
    "legume":    30,   # pulses fix atmospheric N
    "cereal":    0,
    "vegetable": 10,
    "fallow":    0,
    "sugarcane": 15,
}

IRRIGATION_TYPES = ["rainfed", "irrigated", "drip"]

# Irrigation efficiency multiplier on nutrient uptake
IRRIGATION_FACTOR = {
    "rainfed":   0.75,
    "irrigated": 1.00,
    "drip":      1.15,  # drip fertigation is most efficient
}

# Fertilizer type selection logic based on nutrient ratios
def select_fertilizer(N_dose, P_dose, K_dose, soil_pH):
    """
    Returns primary fertilizer recommendation based on dominant need
    and soil pH (affects P availability)
    """
    if N_dose > 100 and P_dose > 50:
        return "DAP + Urea"          # high N+P need
    elif N_dose > 80 and K_dose > 60:
        return "Urea + MOP"          # high N+K need
    elif P_dose > 60 and K_dose > 60:
        return "NPK Complex (10:26:26)"
    elif N_dose > 100:
        return "Urea"
    elif P_dose > 50 and soil_pH > 7.5:
        return "SSP"                 # SSP better in alkaline soils
    elif P_dose > 50:
        return "DAP"
    elif K_dose > 60:
        return "MOP + Urea"
    else:
        return "NPK Complex (12:32:16)"

# ── Generate dataset ──────────────────────────────────────────
records = []

for _ in range(15000):
    crop       = np.random.choice(list(ICAR_NUTRIENTS.keys()))
    soil       = np.random.choice(SOIL_TYPES)
    irrigation = np.random.choice(IRRIGATION_TYPES)
    prev_crop  = np.random.choice(list(PREV_CROP_N_CREDIT.keys()))

    N_req, P_req, K_req, base_yield = ICAR_NUTRIENTS[crop]

    # Soil test values (current nutrient status)
    # Low: <280 N, <11 P, <110 K (kg/ha available)
    # Medium: 280-560 N, 11-22 P, 110-280 K
    # High: >560 N, >22 P, >280 K
    soil_N = np.random.choice(
        [np.random.uniform(50, 280),    # low
         np.random.uniform(280, 560),   # medium
         np.random.uniform(560, 900)],  # high
        p=[0.45, 0.40, 0.15]
    )
    soil_P = np.random.choice(
        [np.random.uniform(2, 11),
         np.random.uniform(11, 22),
         np.random.uniform(22, 45)],
        p=[0.40, 0.40, 0.20]
    )
    soil_K = np.random.choice(
        [np.random.uniform(50, 110),
         np.random.uniform(110, 280),
         np.random.uniform(280, 500)],
        p=[0.35, 0.45, 0.20]
    )

    sf = SOIL_FACTORS[soil]
    pH_lo, pH_hi = sf["pH_range"]
    soil_pH = np.random.uniform(pH_lo, pH_hi)

    # Organic matter affects N availability
    organic_matter = np.random.uniform(0.3, 2.5)  # %

    # Target yield variation (±20% of base)
    target_yield = base_yield * np.random.uniform(0.8, 1.2)

    # ── Core calculation (Nutrient Balance Method) ────────────
    # Dose = (Crop requirement - Soil supply) / Fertilizer efficiency
    # Soil supply = soil_nutrient * availability_factor

    N_supply = (soil_N * 0.02) + (organic_matter * 20)  # kg/ha available
    P_supply = soil_P * 0.5
    K_supply = soil_K * 0.02

    N_credit = PREV_CROP_N_CREDIT[prev_crop]

    irr_factor = IRRIGATION_FACTOR[irrigation]

    # Scale requirement by target yield ratio
    yield_ratio = target_yield / base_yield

    N_dose = max(0, (N_req * yield_ratio - N_supply - N_credit) / sf["N"] / irr_factor)
    P_dose = max(0, (P_req * yield_ratio - P_supply) / sf["P"])
    K_dose = max(0, (K_req * yield_ratio - K_supply) / sf["K"])

    # pH correction — P availability drops sharply above pH 7.5 and below 5.5
    if soil_pH > 7.5:
        P_dose *= 1.25   # need more P in alkaline soils
    elif soil_pH < 5.5:
        P_dose *= 1.15   # need more P in acidic soils

    # Add realistic noise
    N_dose += np.random.normal(0, 5)
    P_dose += np.random.normal(0, 3)
    K_dose += np.random.normal(0, 3)

    N_dose = max(0, round(N_dose, 1))
    P_dose = max(0, round(P_dose, 1))
    K_dose = max(0, round(K_dose, 1))

    fertilizer = select_fertilizer(N_dose, P_dose, K_dose, soil_pH)

    records.append({
        "crop":           crop,
        "soil_type":      soil,
        "irrigation":     irrigation,
        "prev_crop":      prev_crop,
        "soil_N":         round(soil_N, 1),
        "soil_P":         round(soil_P, 2),
        "soil_K":         round(soil_K, 1),
        "soil_pH":        round(soil_pH, 2),
        "organic_matter": round(organic_matter, 2),
        "target_yield":   round(target_yield, 2),
        "N_dose":         N_dose,
        "P_dose":         P_dose,
        "K_dose":         K_dose,
        "fertilizer":     fertilizer,
    })

df = pd.DataFrame(records)
print(f"✅ Dataset: {df.shape[0]} rows, {df['crop'].nunique()} crops, {df['soil_type'].nunique()} soil types")
print(f"   Avg N dose: {df['N_dose'].mean():.1f} kg/ha")
print(f"   Avg P dose: {df['P_dose'].mean():.1f} kg/ha")
print(f"   Avg K dose: {df['K_dose'].mean():.1f} kg/ha")
print(f"   Fertilizer types: {df['fertilizer'].value_counts().to_dict()}\n")

# ── Encode categoricals ───────────────────────────────────────
le_crop  = LabelEncoder()
le_soil  = LabelEncoder()
le_irr   = LabelEncoder()
le_prev  = LabelEncoder()
le_fert  = LabelEncoder()

df["crop_enc"] = le_crop.fit_transform(df["crop"])
df["soil_enc"] = le_soil.fit_transform(df["soil_type"])
df["irr_enc"]  = le_irr.fit_transform(df["irrigation"])
df["prev_enc"] = le_prev.fit_transform(df["prev_crop"])
df["fert_enc"] = le_fert.fit_transform(df["fertilizer"])

features = ["crop_enc", "soil_enc", "irr_enc", "prev_enc",
            "soil_N", "soil_P", "soil_K", "soil_pH",
            "organic_matter", "target_yield"]

X = df[features]

# ── Train 3 regression models (N, P, K doses) ────────────────
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
idx_train = X_train.index
idx_test  = X_test.index

print("⏳ Training N dose model...")
model_N = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model_N.fit(X_train, df.loc[idx_train, "N_dose"])

print("⏳ Training P dose model...")
model_P = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model_P.fit(X_train, df.loc[idx_train, "P_dose"])

print("⏳ Training K dose model...")
model_K = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model_K.fit(X_train, df.loc[idx_train, "K_dose"])

print("⏳ Training fertilizer type classifier...")
model_fert = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model_fert.fit(X_train, df.loc[idx_train, "fert_enc"])

# ── Evaluate ──────────────────────────────────────────────────
print("\n🎯 Model Performance:")
for name, model, col in [("N dose", model_N, "N_dose"), ("P dose", model_P, "P_dose"), ("K dose", model_K, "K_dose")]:
    pred = np.maximum(model.predict(X_test), 0)
    print(f"   {name}: MAE={mean_absolute_error(df.loc[idx_test, col], pred):.2f} kg/ha  R²={r2_score(df.loc[idx_test, col], pred):.4f}")

fert_pred = model_fert.predict(X_test)
print(f"   Fertilizer type: Accuracy={accuracy_score(df.loc[idx_test, 'fert_enc'], fert_pred)*100:.1f}%")

# ── Feature importance ────────────────────────────────────────
feat_names = ["Crop", "Soil", "Irrigation", "Prev Crop", "Soil N", "Soil P", "Soil K", "pH", "Organic Matter", "Target Yield"]
print("\n📌 Feature Importance (N model):")
for name, imp in sorted(zip(feat_names, model_N.feature_importances_), key=lambda x: -x[1]):
    bar = "█" * int(imp * 60)
    print(f"   {name:<16} {imp*100:5.1f}%  {bar}")

# ── Save ──────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

joblib.dump(model_N,    os.path.join(BASE_DIR, "model_N.pkl"))
joblib.dump(model_P,    os.path.join(BASE_DIR, "model_P.pkl"))
joblib.dump(model_K,    os.path.join(BASE_DIR, "model_K.pkl"))
joblib.dump(model_fert, os.path.join(BASE_DIR, "model_fert.pkl"))
joblib.dump(le_crop,    os.path.join(BASE_DIR, "le_crop.pkl"))
joblib.dump(le_soil,    os.path.join(BASE_DIR, "le_soil.pkl"))
joblib.dump(le_irr,     os.path.join(BASE_DIR, "le_irr.pkl"))
joblib.dump(le_prev,    os.path.join(BASE_DIR, "le_prev.pkl"))
joblib.dump(le_fert,    os.path.join(BASE_DIR, "le_fert.pkl"))

print("\n✅ All models saved.")
print("🚀 Run: streamlit run app.py")
