"""Model registry.

Every artifact is loaded ONCE at process start and kept in memory. A bundle
that fails to load does not take the service down - its endpoint reports 503
and everything else keeps serving (Phase 14: graceful degradation).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import joblib

from .config import ARTIFACT_DIR, DISEASE_MODEL_PATH

log = logging.getLogger(__name__)


@dataclass
class Bundle:
    """One model family: its loaded objects plus why it is unavailable."""

    name: str
    objects: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.error is None

    def __getitem__(self, key: str) -> Any:
        return self.objects[key]


_BUNDLES: dict[str, Bundle] = {}

# subdir -> {alias: filename}. Filenames are exactly what train.py wrote.
_JOBLIB_SPEC: dict[str, dict[str, str]] = {
    "crop": {
        "model": "crop_model.pkl",
        "le": "label_encoder.pkl",
    },
    "fertilizer": {
        "model_N": "model_N.pkl",
        "model_P": "model_P.pkl",
        "model_K": "model_K.pkl",
        "model_fert": "model_fert.pkl",
        "le_crop": "le_crop.pkl",
        "le_soil": "le_soil.pkl",
        "le_irr": "le_irr.pkl",
        "le_prev": "le_prev.pkl",
        "le_fert": "le_fert.pkl",
    },
    "water": {
        "model_water": "water_model.pkl",
        "model_interval": "interval_model.pkl",
        "le_crop": "le_crop.pkl",
        "le_soil": "le_soil.pkl",
        "le_stage": "le_stage.pkl",
    },
    "rainfall": {
        "model": "rainfall_model.pkl",
        "le_state": "le_state.pkl",
        "le_season": "le_season.pkl",
    },
}


def _load_joblib_bundle(name: str, spec: dict[str, str]) -> Bundle:
    bundle = Bundle(name=name)
    directory = ARTIFACT_DIR / name
    try:
        for alias, filename in spec.items():
            path = directory / filename
            if not path.exists():
                raise FileNotFoundError(f"missing artifact: {path.name}")
            bundle.objects[alias] = joblib.load(path)
    except Exception as exc:  # noqa: BLE001 - any load failure must be survivable
        bundle.objects.clear()
        bundle.error = f"{type(exc).__name__}: {exc}"
        log.error("model bundle %r failed to load: %s", name, bundle.error)
    else:
        log.info("model bundle %r loaded (%d objects)", name, len(bundle.objects))
    return bundle


def _load_rainfall_normals(bundle: Bundle) -> None:
    """IMD monthly normals ship as JSON next to the rainfall model."""
    if not bundle.ready:
        return
    path = ARTIFACT_DIR / "rainfall" / "imd_normals.json"
    try:
        bundle.objects["normals"] = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        bundle.objects.clear()
        bundle.error = f"{type(exc).__name__}: {exc}"
        log.error("rainfall normals failed to load: %s", bundle.error)


def _load_disease() -> Bundle:
    """Keras classifier. TensorFlow and the .h5 file are both optional."""
    bundle = Bundle(name="disease")
    indices_path = ARTIFACT_DIR / "disease" / "class_indices.json"
    try:
        raw = json.loads(indices_path.read_text(encoding="utf-8"))
        # class_indices.json ships as {"0": "Apple___Apple_scab", ...}
        bundle.objects["class_indices"] = {int(k): v for k, v in raw.items()}
    except Exception as exc:  # noqa: BLE001
        bundle.error = f"class_indices.json unreadable: {exc}"
        return bundle

    if not DISEASE_MODEL_PATH.exists():
        bundle.error = (
            f"model weights not found at {DISEASE_MODEL_PATH}. "
            "The trained Keras file is not distributed with this repository - "
            "see docs/plant-disease-model.md."
        )
        log.warning("disease bundle unavailable: %s", bundle.error)
        return bundle

    try:
        from . import disease_model  # noqa: PLC0415 - pulls in TensorFlow

        bundle.objects["model"] = disease_model.load(DISEASE_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        bundle.error = f"{type(exc).__name__}: {exc}"
        log.error("disease model failed to load: %s", bundle.error)
    return bundle


def load_all() -> None:
    """Called once from the FastAPI lifespan hook."""
    for name, spec in _JOBLIB_SPEC.items():
        _BUNDLES[name] = _load_joblib_bundle(name, spec)
    _load_rainfall_normals(_BUNDLES["rainfall"])
    _BUNDLES["disease"] = _load_disease()


def get(name: str) -> Bundle:
    return _BUNDLES.get(name) or Bundle(name=name, error="registry not initialised")


def status() -> dict[str, dict[str, Any]]:
    return {
        name: {"ready": b.ready, "error": b.error}
        for name, b in sorted(_BUNDLES.items())
    }
