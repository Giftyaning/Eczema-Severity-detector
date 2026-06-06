"""Shared base class for model wrappers.

Each wrapper owns one Keras model, loads its weights once, and exposes a
uniform interface (`preprocess`, `predict`) so the predictor service can
treat CNN and MobileNetV2 identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class LoadedModel:
    """Base wrapper around a trained Keras binary classifier.

    Subclasses must implement ``preprocess`` (turn a raw uint8 RGB array into
    the exact tensor the model was trained on) and may override ``build`` to
    construct an untrained architecture for the training script.
    """

    name: str = "base"

    def __init__(self, weights_path: Path):
        self.weights_path = Path(weights_path)
        self._model = None  # lazy: populated by load()

    # ── lifecycle ────────────────────────────────────────────
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        """The underlying Keras model (Grad-CAM needs direct access)."""
        if self._model is None:
            raise RuntimeError(f"{self.name} model is not loaded.")
        return self._model

    def load(self) -> bool:
        """Load weights from disk. Returns True on success, False if absent.

        Never raises on a missing file — the app can start in a 'degraded'
        state and report it via /health rather than crashing.
        """
        if not self.weights_path.exists():
            return False
        # Imported lazily so modules that only need config/schemas don't pull
        # in TensorFlow.
        import tensorflow as tf

        self._model = tf.keras.models.load_model(self.weights_path)
        return True

    # ── inference ────────────────────────────────────────────
    def preprocess(self, image: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Convert a (H, W, 3) uint8 RGB array into a batched model input."""
        raise NotImplementedError

    def predict(self, image: np.ndarray) -> float:
        """Run inference on a single raw RGB array and return the sigmoid score."""
        batch = self.preprocess(image)
        score = float(self.model.predict(batch, verbose=0)[0][0])
        return score
