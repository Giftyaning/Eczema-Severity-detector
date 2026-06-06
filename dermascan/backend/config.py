"""Central configuration for the DermaScan backend.

Every tunable value lives here. Route handlers, services and model loaders
import from this module rather than hardcoding values, so the whole system
can be retargeted by editing one file.
"""

from __future__ import annotations

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────

# backend/ directory (this file's parent)
BACKEND_DIR = Path(__file__).resolve().parent

# Where trained model weights live. train_and_save.py writes here and the
# model loaders read from here at startup.
MODEL_DIR = BACKEND_DIR / "models" / "weights"

CNN_WEIGHTS_PATH = MODEL_DIR / "cnn.keras"
MOBILENET_WEIGHTS_PATH = MODEL_DIR / "mobilenet.keras"

# Dataset root used only by train_and_save.py (never by the request path).
# Overridable so Docker can mount the dataset elsewhere.
DATASET_DIR = Path(os.getenv("DERMASCAN_DATASET_DIR", BACKEND_DIR.parent.parent / "dataset"))
TRAIN_DIR = DATASET_DIR / "train"
VALIDATION_DIR = DATASET_DIR / "validation"

# ─────────────────────────────────────────────────────────────
# Image / model parameters  (match the original training scripts)
# ─────────────────────────────────────────────────────────────

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
EPOCHS = 10
MOBILENET_LEARNING_RATE = 0.0001

# ─────────────────────────────────────────────────────────────
# Class mapping
# ─────────────────────────────────────────────────────────────
#
# tf.keras.utils.image_dataset_from_directory assigns labels alphabetically,
# so the folders map as:  high -> 0 ,  low -> 1
# A sigmoid output therefore means:  score > THRESHOLD  ==>  index 1 ("low").
# CLASS_NAMES is ordered to match that index mapping exactly.
CLASS_NAMES = ["high", "low"]  # index 0 = high severity, index 1 = low severity
THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────
# Upload validation
# ─────────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

# ─────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────

GRADCAM_OVERLAY_ALPHA = 0.7  # heatmap opacity when blended onto the original
GRADCAM_COLORMAP = "jet"     # blue -> cyan -> green -> yellow -> red

# ─────────────────────────────────────────────────────────────
# App / API
# ─────────────────────────────────────────────────────────────

API_PREFIX = "/api/v1"
APP_VERSION = "1.0.0"

# CORS: allow everything in dev. In production set DERMASCAN_ALLOWED_ORIGINS to
# a comma-separated list (e.g. "https://dermascan.app,https://www.dermascan.app").
_origins_env = os.getenv("DERMASCAN_ALLOWED_ORIGINS", "").strip()
if _origins_env:
    CORS_ALLOW_ORIGINS = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    CORS_ALLOW_ORIGINS = ["*"]
