"""Train the crop recommendation model.

Data sources
------------
1. `data/Crop_recommendation.csv` - the standard Kaggle crop dataset
   (2200 rows, 22 crops). This is the original project's training data.

2. `data/crop_data2.csv` - from KRUTHIKTR/Crop-Recommendation-System-Using-
   Machine-Learning (MIT). Only a small part of it is usable, and this script
   is deliberate about which part.

Why most of dataset 2 is discarded
----------------------------------
1500+ of its feature rows are byte-identical to rows in dataset 1, and 547 of
those carry a DIFFERENT crop label:

    kidneybeans -> "beans"        (125 rows)
    mungbean    -> "cowpeas"      (122 rows)
    chickpea    -> "Soyabeans"    (100 rows)
    pigeonpeas  -> "peas"         (100 rows)
    mothbeans   -> "groundnuts"   (100 rows)

Those are existing pulse crops renamed, not new observations. Training on them
would teach the model that chickpea soil means soybeans, which is wrong advice
for a farmer deciding what to sow, and would also split each pulse's evidence
across two labels. They are excluded by name in RELABELLED_DUPLICATES.

What is kept is the genuinely new material: four crops that appear nowhere in
dataset 1 (arecanut, jackfruit, sugarcane, groundnut), plus a few extra samples
for crops already present.

Run this to regenerate crop_model.pkl and label_encoder.pkl:

    python train_crop.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR.parent / "artifacts" / "crop"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Labels in dataset 2 that are existing crops under a different name. See the
# module docstring - these are excluded, not merged.
RELABELLED_DUPLICATES = {"beans", "cowpeas", "soyabeans", "peas", "groundnuts"}

# Dataset 2 mixes capitalisation ("Arecanut" beside "apple") and misspells
# soybean. Everything is lowercased; this maps what remains onto clean names.
LABEL_FIXES = {"arecanut": "arecanut", "jackfruit": "jackfruit",
               "sugarcane": "sugarcane", "groundnut": "groundnut"}

RANDOM_STATE = 42


def _row_keys(frame: pd.DataFrame) -> list[tuple]:
    """Feature tuples rounded so the two files' differing precision matches."""
    return list(map(tuple, frame[FEATURES].round(4).values.tolist()))


def build_dataset() -> pd.DataFrame:
    primary = pd.read_csv(DATA_DIR / "Crop_recommendation.csv")
    primary["label"] = primary["label"].str.strip().str.lower()
    print(f"dataset 1: {len(primary)} rows, {primary.label.nunique()} crops")

    secondary_path = DATA_DIR / "crop_data2.csv"
    if not secondary_path.exists():
        print(f"dataset 2 not found at {secondary_path} - training on dataset 1 only")
        return primary

    secondary = pd.read_csv(secondary_path)
    secondary["label"] = secondary["label"].str.strip().str.lower()
    print(f"dataset 2: {len(secondary)} rows, {secondary.label.nunique()} crops")

    known = set(_row_keys(primary))
    secondary["_key"] = _row_keys(secondary)

    fresh = secondary[~secondary["_key"].isin(known)]
    print(f"  {len(secondary) - len(fresh)} rows already present in dataset 1 - dropped")

    renamed = fresh[fresh.label.isin(RELABELLED_DUPLICATES)]
    if len(renamed):
        print(f"  {len(renamed)} rows with relabelled-duplicate crop names "
              f"({', '.join(sorted(renamed.label.unique()))}) - dropped")
    fresh = fresh[~fresh.label.isin(RELABELLED_DUPLICATES)]

    fresh = fresh.assign(label=fresh.label.map(lambda v: LABEL_FIXES.get(v, v)))
    print(f"  {len(fresh)} rows kept: "
          + ", ".join(f"{k} x{v}" for k, v in fresh.label.value_counts().items()))

    combined = pd.concat([primary, fresh[FEATURES + ["label"]]], ignore_index=True)
    print(f"\ncombined: {len(combined)} rows, {combined.label.nunique()} crops")
    return combined


def main() -> None:
    df = build_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(df["label"])
    X = df[FEATURES].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    accuracy = accuracy_score(y_test, predicted)
    print(f"\nheld-out accuracy: {accuracy * 100:.2f}%\n")
    print(classification_report(y_test, predicted,
                                target_names=encoder.classes_, zero_division=0))

    # RandomForest is kept over the SVC that dataset 2's project publishes:
    # it scores higher on identical data and, unlike that SVC
    # (probability=False), it supports predict_proba - which the UI needs for
    # the confidence score and the top-3 alternatives.
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("crop_model.pkl", "label_encoder.pkl"):
        existing = ARTIFACT_DIR / name
        if existing.exists():
            shutil.copy2(existing, existing.with_suffix(".pkl.bak"))

    joblib.dump(model, ARTIFACT_DIR / "crop_model.pkl")
    joblib.dump(encoder, ARTIFACT_DIR / "label_encoder.pkl")
    print(f"saved to {ARTIFACT_DIR} (previous versions kept as .pkl.bak)")
    print(f"crops: {', '.join(encoder.classes_)}")


if __name__ == "__main__":
    main()
