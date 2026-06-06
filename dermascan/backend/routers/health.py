"""Health endpoint: reports model load status, version and uptime."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from config import APP_VERSION
from schemas.response import HealthResponse
from services.predictor import registry

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(request: Request) -> HealthResponse:
    status_map = registry.status
    all_loaded = all(status_map.values())
    started_at = getattr(request.app.state, "started_at", time.time())
    return HealthResponse(
        status="ok" if all_loaded else "degraded",
        version=APP_VERSION,
        uptime_seconds=round(time.time() - started_at, 3),
        models_loaded=status_map,
    )
