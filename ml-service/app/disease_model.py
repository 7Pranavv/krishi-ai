"""Loading the plant-disease classifier.

The distributed weights are `plant_disease_prediction.h5`, saved by **Keras
3.5.0**. Current Keras (3.15) cannot deserialise that file directly - the
nested Xception's build config trips a shape parser regression:

    ValueError: Cannot convert '((None, 2048),)' to a shape

Keras 2 (`tf_keras`) cannot read it either, because Keras 3 wrote
`batch_shape` where Keras 2 expects `batch_input_shape`. So the file is
loadable by neither current major version.

The way through is to skip the serialised config entirely: rebuild the
architecture from the training notebook and load only the weight arrays,
matched by layer name. That is version-independent.

Verified against the original app's own published screenshot, which shows
`Grape___Black_rot` at 100% for a specific leaf; this loader reproduces that
exactly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

IMG_SIZE = 224
NUM_CLASSES = 38


def build_architecture() -> Any:
    """The notebook's model: Xception (avg-pooled) + BN + 256 + dropout + 38.

    Layer names are pinned to the ones inside the .h5 so `load_weights` can
    match them; renaming any of these silently breaks weight loading.
    """
    import keras
    from keras import layers

    base = keras.applications.Xception(
        weights=None,  # the trained weights arrive via load_weights
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
        name="xception",
    )
    return keras.Sequential(
        [
            keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
            base,
            layers.BatchNormalization(name="batch_normalization_4"),
            layers.Dense(256, activation="relu", name="dense"),
            layers.Dropout(0.5, name="dropout"),
            layers.Dense(NUM_CLASSES, activation="softmax", name="dense_1"),
        ],
        name="sequential",
    )


def load(path: Path) -> Any:
    """Load the classifier from a .keras or .h5 file.

    A native `.keras` file loads directly. An `.h5` goes through the rebuild
    path described in the module docstring.
    """
    import keras

    if path.suffix == ".keras":
        model = keras.saving.load_model(path)
        log.info("disease model loaded from native format: %s", path.name)
        return model

    model = build_architecture()
    # Sanity check that names actually matched: if nothing changed, the file's
    # groups did not line up with this architecture and every prediction would
    # be random noise from an untrained head.
    before = model.layers[-1].get_weights()[0].copy()
    model.load_weights(path)
    after = model.layers[-1].get_weights()[0]
    if (before == after).all():
        raise ValueError(
            f"{path.name} contained no weights matching this architecture - "
            "the output layer is still randomly initialised. The file is "
            "probably a different model; see docs/plant-disease-model.md."
        )
    log.info("disease model rebuilt and weights loaded from %s", path.name)
    return model
