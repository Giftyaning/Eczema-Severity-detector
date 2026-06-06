"""Tests for POST /api/v1/predict and /predict/compare.

Validation tests (bad type, no file) run without trained weights. Inference
tests skip gracefully when weights are absent.
"""

from __future__ import annotations

from conftest import require_models

PREDICT = "/api/v1/predict"
COMPARE = "/api/v1/predict/compare"


# ── Validation (no models needed) ────────────────────────────


def test_predict_no_file_returns_400(client):
    res = client.post(PREDICT, data={"model": "both"})
    assert res.status_code == 400


def test_predict_non_image_returns_400(client):
    files = {"image": ("notes.txt", b"this is not an image", "text/plain")}
    res = client.post(PREDICT, files=files)
    assert res.status_code == 400


def test_predict_invalid_model_returns_400(client, jpeg_bytes):
    files = {"image": ("skin.jpg", jpeg_bytes, "image/jpeg")}
    res = client.post(PREDICT, files=files, data={"model": "resnet"})
    assert res.status_code == 400


# ── Inference (need trained weights) ─────────────────────────


def test_predict_valid_jpeg_returns_schema(client, jpeg_bytes, models_status):
    require_models(models_status, "cnn", "mobilenet")
    files = {"image": ("skin.jpg", jpeg_bytes, "image/jpeg")}
    res = client.post(PREDICT, files=files, data={"model": "both", "gradcam": "true"})
    assert res.status_code == 200

    body = res.json()
    assert body["status"] == "success"
    assert "processing_time_ms" in body and body["processing_time_ms"] >= 0

    preds = body["predictions"]
    for name in ("cnn", "mobilenet"):
        assert name in preds
        p = preds[name]
        assert p["label"] in {"high", "low"}
        assert 0.0 <= p["confidence"] <= 1.0
        assert 0.0 <= p["raw_score"] <= 1.0

    # agreement must reflect the labels, not be hardcoded.
    assert body["agreement"] == (preds["cnn"]["label"] == preds["mobilenet"]["label"])
    # gradcam requested -> overlay present.
    assert isinstance(body["gradcam_base64"], str) and body["gradcam_base64"]


def test_predict_cnn_only_omits_mobilenet(client, jpeg_bytes, models_status):
    require_models(models_status, "cnn")
    files = {"image": ("skin.jpg", jpeg_bytes, "image/jpeg")}
    res = client.post(PREDICT, files=files, data={"model": "cnn", "gradcam": "false"})
    assert res.status_code == 200

    body = res.json()
    assert "cnn" in body["predictions"]
    assert "mobilenet" not in body["predictions"]  # excluded, not null
    # Single model -> no agreement comparison.
    assert "agreement" not in body or body.get("agreement") is None


def test_compare_returns_both_and_delta(client, jpeg_bytes, models_status):
    require_models(models_status, "cnn", "mobilenet")
    files = {
        "image_a": ("a.jpg", jpeg_bytes, "image/jpeg"),
        "image_b": ("b.jpg", jpeg_bytes, "image/jpeg"),
    }
    res = client.post(COMPARE, files=files, data={"model": "both"})
    assert res.status_code == 200

    body = res.json()
    assert body["status"] == "success"
    assert "image_a" in body and "image_b" in body
    assert "predictions" in body["image_a"] and "predictions" in body["image_b"]

    delta = body["severity_delta"]
    assert delta["worse_image"] in {"image_a", "image_b", "tie"}
    assert 0.0 <= delta["delta"] <= 1.0
    # Identical images -> tie, zero delta.
    assert delta["worse_image"] == "tie"
    assert delta["delta"] == 0.0


def test_compare_no_file_returns_400(client, jpeg_bytes):
    files = {"image_a": ("a.jpg", jpeg_bytes, "image/jpeg")}  # missing image_b
    res = client.post(COMPARE, files=files, data={"model": "both"})
    assert res.status_code == 400
