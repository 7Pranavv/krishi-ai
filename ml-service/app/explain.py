"""Why the crop model chose what it chose.

SHAP attributes a prediction to its input features, so a farmer can see that a
recommendation rested on, say, high rainfall and low pH rather than having to
take it on trust. Approach ported from AhqafCoder/AICropRecommendation (MIT).

The explainer is built once against the loaded model and reused - constructing
a TreeExplainer walks every tree in the forest and is far too slow to do per
request.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

FEATURE_LABELS = {
    "N": "Nitrogen", "P": "Phosphorus", "K": "Potassium",
    "temperature": "Temperature", "humidity": "Humidity",
    "ph": "Soil pH", "rainfall": "Rainfall",
}
FEATURE_UNITS = {
    "N": "kg/ha", "P": "kg/ha", "K": "kg/ha",
    "temperature": "°C", "humidity": "%", "ph": "", "rainfall": "mm",
}

_explainer: Any = None
_lock = threading.Lock()
_unavailable: str | None = None


def _get_explainer(model: Any) -> Any:
    """Build the TreeExplainer once. Returns None if SHAP is unusable."""
    global _explainer, _unavailable
    if _explainer is not None or _unavailable:
        return _explainer
    with _lock:
        if _explainer is not None or _unavailable:
            return _explainer
        try:
            import shap  # noqa: PLC0415 - optional, and slow to import

            _explainer = shap.TreeExplainer(model)
            log.info("SHAP explainer ready for the crop model")
        except Exception as exc:  # noqa: BLE001
            _unavailable = str(exc)
            log.warning("SHAP unavailable, explanations disabled: %s", exc)
    return _explainer


def explain_crop(model: Any, features: np.ndarray, feature_names: list[str],
                 class_index: int, top_n: int = 4) -> list[dict] | None:
    """Per-feature contributions to the predicted class.

    Returns entries ordered by influence, each with the input value and whether
    it pushed the model toward or away from this crop. Returns None rather than
    a guess when SHAP is not available - the caller omits the section.
    """
    explainer = _get_explainer(model)
    if explainer is None:
        return None

    try:
        values = explainer.shap_values(features, check_additivity=False)
    except Exception as exc:  # noqa: BLE001
        log.warning("SHAP computation failed: %s", exc)
        return None

    # Shape varies by SHAP version and model: (samples, features, classes) for
    # newer multiclass output, or a per-class list of (samples, features).
    array = np.asarray(values)
    if array.ndim == 3:
        contributions = array[0, :, class_index]
    elif isinstance(values, list):
        contributions = np.asarray(values[class_index])[0]
    else:
        contributions = array[0]

    contributions = np.asarray(contributions, dtype=float).ravel()
    if contributions.shape[0] != len(feature_names):
        log.warning("SHAP returned %d values for %d features; skipping",
                    contributions.shape[0], len(feature_names))
        return None

    raw = features[0]
    order = np.argsort(np.abs(contributions))[::-1][:top_n]
    return [
        {
            "feature": feature_names[i],
            "label": FEATURE_LABELS.get(feature_names[i], feature_names[i]),
            "value": round(float(raw[i]), 2),
            "unit": FEATURE_UNITS.get(feature_names[i], ""),
            "impact": round(float(contributions[i]), 4),
            "direction": "supports" if contributions[i] >= 0 else "counts against",
        }
        for i in order
    ]
