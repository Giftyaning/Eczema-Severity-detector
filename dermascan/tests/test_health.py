"""Tests for GET /api/v1/health."""

from __future__ import annotations


def test_health_returns_200_with_schema(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200

    body = res.json()
    assert set(body) == {"status", "version", "uptime_seconds", "models_loaded"}
    assert body["status"] in {"ok", "degraded"}
    assert body["version"] == "1.0.0"
    assert isinstance(body["uptime_seconds"], (int, float))
    assert body["uptime_seconds"] >= 0
    assert set(body["models_loaded"]) == {"cnn", "mobilenet"}
    assert all(isinstance(v, bool) for v in body["models_loaded"].values())


def test_both_models_load_at_startup(client, models_status):
    """Both models should report loaded once weights have been trained.

    Skips (rather than fails) when weights are absent so the suite is green on a
    fresh checkout; train_and_save.py produces the weights.
    """
    import pytest

    if not all(models_status.values()):
        pytest.skip(
            "Weights not present — run backend/train_and_save.py. "
            f"Status: {models_status}"
        )
    assert models_status == {"cnn": True, "mobilenet": True}
