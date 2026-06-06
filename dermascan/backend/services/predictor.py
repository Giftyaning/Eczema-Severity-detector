"""Prediction orchestration.

Holds the model registry (populated once at startup), validates and decodes
uploaded images entirely in memory, runs the requested model(s), optionally
generates a Grad-CAM overlay, and assembles the typed response.
"""

from __future__ import annotations

import io
import time
from typing import Optional

import numpy as np

from config import CLASS_NAMES, IMG_HEIGHT, IMG_WIDTH, THRESHOLD
from models import CNNModel, MobileNetModel
from schemas.response import (
    CompareResponse,
    ImagePrediction,
    ModelPrediction,
    PredictResponse,
    Predictions,
    SeverityDelta,
)

from . import gradcam

# Index of the HIGH-severity class within CLASS_NAMES (used for severity scoring).
_HIGH_INDEX = CLASS_NAMES.index("high")


class CorruptImageError(Exception):
    """Raised when an uploaded file cannot be decoded as an image."""


class ModelRegistry:
    """Owns the two model wrappers. Loaded once via ``load_all`` at startup."""

    def __init__(self):
        self.cnn = CNNModel()
        self.mobilenet = MobileNetModel()

    def load_all(self) -> dict[str, bool]:
        """Load every model's weights. Returns per-model success flags."""
        return {"cnn": self.cnn.load(), "mobilenet": self.mobilenet.load()}

    @property
    def status(self) -> dict[str, bool]:
        return {"cnn": self.cnn.is_loaded, "mobilenet": self.mobilenet.is_loaded}

    def get(self, name: str):
        return {"cnn": self.cnn, "mobilenet": self.mobilenet}[name]


# Module-level singleton; main.py calls registry.load_all() on startup.
registry = ModelRegistry()


# ─────────────────────────────────────────────────────────────
# Image handling (in memory only — nothing is written to disk)
# ─────────────────────────────────────────────────────────────


def decode_image(raw: bytes) -> np.ndarray:
    """Decode raw bytes into a (H, W, 3) uint8 RGB array resized to model input.

    Raises CorruptImageError if the bytes are not a valid image.
    """
    from PIL import Image, UnidentifiedImageError

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()  # force decode now so corruption surfaces here
        img = img.convert("RGB").resize((IMG_WIDTH, IMG_HEIGHT))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise CorruptImageError(str(exc)) from exc
    return np.asarray(img, dtype="uint8")


# ─────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────


def _interpret(score: float) -> ModelPrediction:
    """Turn a raw sigmoid score into a typed prediction.

    score > THRESHOLD -> index 1 ("low"); otherwise index 0 ("high").
    confidence is the probability of whichever class was chosen.
    """
    index = 1 if score > THRESHOLD else 0
    label = CLASS_NAMES[index]
    confidence = score if index == 1 else 1.0 - score
    return ModelPrediction(label=label, confidence=round(confidence, 4), raw_score=round(score, 4))


def _p_high(score: float) -> float:
    """Probability of HIGH severity from a raw sigmoid score."""
    # index 0 == high, so P(high) = 1 - P(index 1) when high is index 0.
    return (1.0 - score) if _HIGH_INDEX == 0 else score


def _run_models(image_rgb: np.ndarray, which: str) -> tuple[Predictions, list[str]]:
    """Run the requested model(s); return predictions and the names that ran."""
    names = ["cnn", "mobilenet"] if which == "both" else [which]
    preds = Predictions()
    for name in names:
        wrapper = registry.get(name)
        score = wrapper.predict(image_rgb)
        setattr(preds, name, _interpret(score))
    return preds, names


def _agreement(preds: Predictions, names: list[str]) -> Optional[bool]:
    """True iff both models ran and returned the same label; else None."""
    if len(names) < 2:
        return None
    return preds.cnn.label == preds.mobilenet.label


def _aggregate_p_high(preds: Predictions, names: list[str]) -> float:
    """Mean P(high) across the models that ran — used to rank severity."""
    scores = [_p_high(getattr(preds, n).raw_score) for n in names]
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────


def predict(image_rgb: np.ndarray, which: str = "both", want_gradcam: bool = True) -> PredictResponse:
    """Run prediction (+ optional Grad-CAM) for a single image."""
    start = time.perf_counter()

    preds, names = _run_models(image_rgb, which)
    agreement = _agreement(preds, names)

    overlay_b64: Optional[str] = None
    heatmap_b64: Optional[str] = None
    if want_gradcam:
        # Prefer MobileNetV2 for the heatmap when available, else the CNN.
        primary = "mobilenet" if "mobilenet" in names else names[0]
        wrapper = registry.get(primary)
        heatmap_b64, overlay_b64 = gradcam.generate(wrapper.model, wrapper, image_rgb)

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return PredictResponse(
        predictions=preds,
        agreement=agreement,
        gradcam_base64=overlay_b64,
        gradcam_heatmap_base64=heatmap_b64,
        processing_time_ms=elapsed_ms,
    )


def compare(image_a: np.ndarray, image_b: np.ndarray, which: str = "both") -> CompareResponse:
    """Predict both images and report which is the more severe, and by how much."""
    start = time.perf_counter()

    results = []
    for img in (image_a, image_b):
        preds, names = _run_models(img, which)
        results.append(
            ImagePrediction(
                predictions=preds,
                agreement=_agreement(preds, names),
                severity_score=round(_aggregate_p_high(preds, names), 4),
            )
        )

    a, b = results
    delta = round(abs(a.severity_score - b.severity_score), 4)
    if a.severity_score > b.severity_score:
        worse = "image_a"
    elif b.severity_score > a.severity_score:
        worse = "image_b"
    else:
        worse = "tie"

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return CompareResponse(
        image_a=a,
        image_b=b,
        severity_delta=SeverityDelta(worse_image=worse, delta=delta),
        processing_time_ms=elapsed_ms,
    )
