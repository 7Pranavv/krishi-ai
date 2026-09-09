"""Shared helpers for the prediction routers."""
from __future__ import annotations

from fastapi import HTTPException, status

from . import registry


def require_bundle(name: str) -> registry.Bundle:
    """Return a loaded bundle or fail with a 503 the caller can degrade on."""
    bundle = registry.get(name)
    if not bundle.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": f"The '{name}' model is not available.",
                "detail": bundle.error,
            },
        )
    return bundle


def encode(label_encoder, value: str, field: str) -> int:
    """Map a categorical value through a trained LabelEncoder.

    Unknown values become a 422 listing what the model actually accepts,
    instead of the ValueError sklearn would raise.
    """
    classes = [str(c) for c in label_encoder.classes_]
    if value not in classes:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": f"Unsupported value for '{field}': {value!r}",
                "detail": f"Allowed values: {', '.join(sorted(classes))}",
            },
        )
    return int(label_encoder.transform([value])[0])
