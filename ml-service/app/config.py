"""Configuration for the Krishi.AI ML service."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Where the .pkl / .h5 / .json artifacts live. Override for containers or a
# mounted model volume.
ARTIFACT_DIR = Path(os.getenv("ML_ARTIFACT_DIR", BASE_DIR / "artifacts"))

# The Keras plant-disease model is ~245 MB and is NOT shipped in this repo.
# Set DISEASE_MODEL_PATH to point at it, or drop it in artifacts/disease/ under
# either name below. The native .keras format is preferred because the original
# .h5 needs an architecture rebuild to load - see disease_model.py.
_DISEASE_DIR = ARTIFACT_DIR / "disease"
_DISEASE_CANDIDATES = (
    _DISEASE_DIR / "plant_disease_prediction.keras",
    _DISEASE_DIR / "plant_disease_prediction.h5",
)


def _resolve_disease_model() -> Path:
    override = os.getenv("DISEASE_MODEL_PATH")
    if override:
        return Path(override)
    for candidate in _DISEASE_CANDIDATES:
        if candidate.exists():
            return candidate
    return _DISEASE_CANDIDATES[-1]  # reported in the "not found" message


DISEASE_MODEL_PATH = _resolve_disease_model()

# Reject anything bigger before it reaches Pillow.
MAX_IMAGE_BYTES = int(os.getenv("ML_MAX_IMAGE_BYTES", 8 * 1024 * 1024))

# Below this the answer is reported but flagged as unreliable.
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("ML_LOW_CONFIDENCE", 0.55))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
