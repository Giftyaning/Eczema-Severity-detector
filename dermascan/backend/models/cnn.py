"""Custom CNN: architecture, loader and inference.

Architecture is identical to the original ``cnn_model.py`` (3x Conv/Pool +
Dense head). The CNN expects pixels normalised to [0, 1] (divide by 255).
"""

from __future__ import annotations

import numpy as np

from config import CNN_WEIGHTS_PATH, IMG_HEIGHT, IMG_WIDTH

from ._base import LoadedModel


def build_cnn():
    """Build the untrained custom CNN. Used by train_and_save.py.

    Mirrors the original cnn_model.py layer-for-layer so saved weights and
    Grad-CAM behaviour match the report.
    """
    import tensorflow as tf

    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
            ),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="dermascan_cnn",
    )


class CNNModel(LoadedModel):
    """Loader + inference wrapper for the custom CNN."""

    name = "cnn"

    def __init__(self):
        super().__init__(CNN_WEIGHTS_PATH)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Normalise pixels to [0, 1] and add a batch dimension."""
        arr = image.astype("float32") / 255.0
        return np.expand_dims(arr, axis=0)
