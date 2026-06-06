"""MobileNetV2 transfer model: architecture, loader and inference.

Architecture is identical to the original ``mobilenet_model.py`` (frozen
ImageNet base + GAP + Dense head). MobileNetV2 requires its own
``preprocess_input`` which scales pixels to [-1, 1].
"""

from __future__ import annotations

import numpy as np

from config import IMG_HEIGHT, IMG_WIDTH, MOBILENET_LEARNING_RATE, MOBILENET_WEIGHTS_PATH

from ._base import LoadedModel


def build_mobilenet():
    """Build and compile the untrained MobileNetV2 transfer model.

    Used by train_and_save.py. Mirrors the original mobilenet_model.py.
    """
    import tensorflow as tf

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="dermascan_mobilenet",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=MOBILENET_LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class MobileNetModel(LoadedModel):
    """Loader + inference wrapper for MobileNetV2."""

    name = "mobilenet"

    def __init__(self):
        super().__init__(MOBILENET_WEIGHTS_PATH)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply MobileNetV2 preprocessing (scales to [-1, 1]) and batch."""
        import tensorflow as tf

        arr = image.astype("float32")
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
        return np.expand_dims(arr, axis=0)
