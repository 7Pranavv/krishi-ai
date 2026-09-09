"""Grad-CAM heatmaps for the disease classifier.

Shows which part of the leaf drove the prediction. A confident label on its own
is not inspectable - if the model fixated on a shadow or the background, the
farmer has no way to tell. The overlay makes that visible.

Method: gradients of the predicted class with respect to the last convolutional
feature map, averaged per channel, used to weight that map.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Any

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

# Final activation of the Xception backbone, before global average pooling.
LAST_CONV_LAYER = "block14_sepconv2_act"

# Contrast curve for the upsampled heatmap; >1 concentrates on the hottest area.
GAMMA = 2.0

_grad_model: Any = None
_unavailable: str | None = None


def _build_grad_model(model: Any) -> Any:
    """A model returning (last conv feature map, pooled features).

    The classifier head is applied separately so gradients can be traced back
    to the convolutional output.
    """
    global _grad_model, _unavailable
    if _grad_model is not None or _unavailable:
        return _grad_model
    try:
        import keras

        base = model.layers[0]  # the nested Xception functional model
        conv_layer = base.get_layer(LAST_CONV_LAYER)
        _grad_model = keras.Model(base.inputs, [conv_layer.output, base.output])
    except Exception as exc:  # noqa: BLE001
        _unavailable = str(exc)
        log.warning("Grad-CAM unavailable: %s", exc)
    return _grad_model


def _colourise(heatmap: np.ndarray) -> np.ndarray:
    """Map [0,1] intensities to a blue-green-yellow-red ramp.

    Hand-rolled rather than pulling in matplotlib for one colormap.
    """
    h = np.clip(heatmap, 0.0, 1.0)
    red = np.clip(1.5 - np.abs(4.0 * h - 3.0), 0, 1)
    green = np.clip(1.5 - np.abs(4.0 * h - 2.0), 0, 1)
    blue = np.clip(1.5 - np.abs(4.0 * h - 1.0), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype("uint8")


def heatmap_overlay(model: Any, tensor: np.ndarray, image: Image.Image,
                    class_index: int, opacity: float = 0.45) -> str | None:
    """Return a data-URL PNG of the leaf with the heatmap overlaid.

    None when Grad-CAM cannot run - the caller simply omits the image rather
    than showing something misleading.
    """
    grad_model = _build_grad_model(model)
    if grad_model is None:
        return None

    try:
        import tensorflow as tf

        head = model.layers[1:]  # BatchNorm -> Dense -> Dropout -> Dense
        final = head[-1]
        inputs = tf.convert_to_tensor(tensor)

        with tf.GradientTape() as tape:
            conv_output, pooled = grad_model(inputs, training=False)
            tape.watch(conv_output)
            activations = pooled
            for layer in head[:-1]:
                activations = layer(activations, training=False)
            # Differentiate the pre-softmax logit, not the probability. This
            # model is confident enough to output an exact 1.0, where softmax
            # has zero gradient and every heatmap would come out blank.
            logits = tf.matmul(activations, final.kernel) + final.bias
            score = logits[:, class_index]

        gradients = tape.gradient(score, conv_output)
        if gradients is None:
            log.warning("Grad-CAM produced no gradients")
            return None

        weights = tf.reduce_mean(gradients, axis=(0, 1, 2))
        cam = tf.reduce_sum(conv_output[0] * weights, axis=-1).numpy()

        cam = np.maximum(cam, 0)
        peak = cam.max()
        if peak <= 0:
            return None  # nothing positively supported the class
        cam /= peak
        # The feature map is only 7x7 and this model attends to most of the
        # leaf, so a linear ramp tints the whole photo evenly and shows little.
        # The gamma pushes mid-range values down and keeps the hottest region
        # distinct.
        cam = cam ** GAMMA

        # Upsample the 7x7 map to the photo, then alpha-blend per pixel so cool
        # areas stay as the original image.
        photo = np.asarray(image.convert("RGB"), dtype="float32")
        height, width = photo.shape[:2]
        upscaled = np.asarray(
            Image.fromarray((cam * 255).astype("uint8")).resize((width, height), Image.BICUBIC),
            dtype="float32",
        ) / 255.0
        ramp = _colourise(upscaled).astype("float32")
        alpha = (upscaled * opacity)[..., None]
        blended = photo * (1.0 - alpha) + ramp * alpha

        buffer = io.BytesIO()
        Image.fromarray(blended.astype("uint8")).save(buffer, format="PNG", optimize=True)
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception as exc:  # noqa: BLE001
        log.warning("Grad-CAM failed: %s", exc)
        return None
