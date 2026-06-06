"""Shared pytest fixtures.

Adds the backend package to sys.path (so `import main` works the same way
`uvicorn main:app` does), provides a TestClient, an in-memory JPEG generator,
and a helper to skip model-dependent tests when weights haven't been trained.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import pytest

# Make backend modules importable as top-level (config, main, services, ...).
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))


@pytest.fixture(scope="session")
def client():
    """A TestClient bound to the FastAPI app (runs lifespan -> loads models)."""
    from fastapi.testclient import TestClient

    import main

    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def jpeg_bytes():
    """Return a small valid JPEG as raw bytes."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (180, 110, 90)).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def models_status():
    """Per-model load status from the live registry."""
    from services.predictor import registry

    return registry.status


def require_models(status: dict, *names: str):
    """Skip the calling test unless every named model is loaded."""
    missing = [n for n in names if not status.get(n, False)]
    if missing:
        pytest.skip(
            f"Model(s) not loaded: {', '.join(missing)}. "
            "Run `python backend/train_and_save.py` to enable inference tests."
        )
