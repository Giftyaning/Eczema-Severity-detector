"""DermaScan FastAPI application entry point.

Both models are loaded once at startup via a lifespan handler — never per
request. CORS is open in development and restrictable via an env var in
production. Run with:

    cd dermascan/backend
    uvicorn main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import API_PREFIX, APP_VERSION, CORS_ALLOW_ORIGINS
from routers import health, predict
from services.predictor import registry


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model weights once when the server starts."""
    app.state.started_at = time.time()
    loaded = registry.load_all()
    for name, ok in loaded.items():
        print(f"[startup] {name}: {'loaded' if ok else 'NOT FOUND (run train_and_save.py)'}")
    yield
    # Nothing to tear down — Keras models are released with the process.


app = FastAPI(
    title="DermaScan API",
    version=APP_VERSION,
    description="Eczema severity classification (high/low) with CNN + MobileNetV2 and Grad-CAM.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Versioned API routes.
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(predict.router, prefix=API_PREFIX)


@app.get("/", tags=["meta"])
def root():
    return {"name": "DermaScan API", "version": APP_VERSION, "docs": "/docs", "api": API_PREFIX}
