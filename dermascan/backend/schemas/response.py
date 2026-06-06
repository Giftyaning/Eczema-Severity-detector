"""Pydantic response models for the DermaScan API.

All response bodies are typed — no raw dicts are returned from handlers.
The shapes here are the single source of truth for the API contract and are
what FastAPI uses to generate the OpenAPI docs.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

# Severity labels are constrained to the configured class names.
SeverityLabel = Literal["high", "low"]


class ModelPrediction(BaseModel):
    """A single model's verdict for one image."""

    label: SeverityLabel = Field(..., description="Predicted severity class.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of the predicted class (0-1).",
    )
    raw_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw sigmoid output. >0.5 maps to 'low', <=0.5 maps to 'high'.",
    )

    model_config = {
        "json_schema_extra": {
            "example": {"label": "high", "confidence": 0.79, "raw_score": 0.21}
        }
    }


class Predictions(BaseModel):
    """Container for each model's prediction.

    Either key may be omitted when the caller requests only one model
    (e.g. model="cnn" omits the mobilenet key entirely).
    """

    cnn: Optional[ModelPrediction] = None
    mobilenet: Optional[ModelPrediction] = None


class PredictResponse(BaseModel):
    """Full response for POST /predict."""

    status: Literal["success"] = "success"
    predictions: Predictions
    agreement: Optional[bool] = Field(
        None,
        description=(
            "True when both models returned the same label. Null when only "
            "one model was requested (nothing to compare)."
        ),
    )
    gradcam_base64: Optional[str] = Field(
        None,
        description="Base64-encoded PNG overlay (heatmap blended on original), or null.",
    )
    gradcam_heatmap_base64: Optional[str] = Field(
        None,
        description=(
            "Base64-encoded PNG of the raw jet-colormap heatmap (no blending), or null. "
            "Optional companion to gradcam_base64 used by the three-panel results view."
        ),
    )
    processing_time_ms: int = Field(..., ge=0)

    model_config = {
        "json_schema_extra": {
            "example": {
                "status": "success",
                "predictions": {
                    "cnn": {"label": "high", "confidence": 0.79, "raw_score": 0.21},
                    "mobilenet": {"label": "high", "confidence": 0.84, "raw_score": 0.16},
                },
                "agreement": True,
                "gradcam_base64": "iVBORw0KGgo...",
                "processing_time_ms": 1204,
            }
        }
    }


# ─────────────────────────────────────────────────────────────
# Compare endpoint
# ─────────────────────────────────────────────────────────────


class ImagePrediction(BaseModel):
    """Predictions for one of the two images in a comparison."""

    predictions: Predictions
    agreement: Optional[bool] = None
    # Mean of the available models' P(high) — used to rank severity.
    severity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Aggregate probability of HIGH severity across the requested models.",
    )


class SeverityDelta(BaseModel):
    """Which image is worse, and by how much."""

    worse_image: Literal["image_a", "image_b", "tie"] = Field(
        ..., description="Which image has the higher aggregate HIGH-severity score."
    )
    delta: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Absolute difference in HIGH-severity score between the two images.",
    )


class CompareResponse(BaseModel):
    """Full response for POST /predict/compare."""

    status: Literal["success"] = "success"
    image_a: ImagePrediction
    image_b: ImagePrediction
    severity_delta: SeverityDelta
    processing_time_ms: int = Field(..., ge=0)


# ─────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Full response for GET /health."""

    status: Literal["ok", "degraded"] = Field(
        ..., description="'ok' when all requested models are loaded, else 'degraded'."
    )
    version: str
    uptime_seconds: float = Field(..., ge=0.0)
    models_loaded: dict[str, bool] = Field(
        ..., description="Per-model load status, e.g. {'cnn': true, 'mobilenet': true}."
    )


# ─────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Uniform error envelope."""

    status: Literal["error"] = "error"
    detail: str
