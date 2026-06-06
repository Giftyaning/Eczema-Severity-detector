# DermaScan

Binary eczema-severity classification (**high** vs **low**) that runs a custom CNN and a transfer-learned MobileNetV2 side by side and explains each prediction with a Grad-CAM heatmap, served as a FastAPI backend with a dark-themed static frontend.

> ⚠️ **Not a medical device.** Educational project only and not validated for clinical use.

---

## Architecture

```
                        ┌──────────────────────────────────────────────┐
   Browser              │                 Backend (FastAPI)             │
 ┌──────────┐  upload   │  routers/predict.py   ── validate type/size   │
 │ frontend │ ────────► │        │              (in memory, never disk) │
 │ (static) │  JSON     │        ▼                                      │
 │  nginx   │ ◄──────── │  services/predictor.py                        │
 └──────────┘           │        │  ┌── models/cnn.py        (÷255)      │
   :3000                │        ├─►│   models/mobilenet.py  (preproc.)  │
                        │        │  └── loaded ONCE at startup (lifespan)│
                        │        ▼                                      │
                        │  services/gradcam.py  ── last Conv2D (auto)   │
                        │                          jet heatmap + overlay│
                        └──────────────────────────────────────────────┘
                                          :8000  /api/v1
```

- **Two models, one verdict.** Custom CNN (3× Conv/Pool + dense head, pixels ÷255) and MobileNetV2 (frozen ImageNet base + GAP + dense head, `preprocess_input` → [-1, 1]). Both take `224×224×3`.
- **`agreement`** is computed from the two labels — never hardcoded.
- **Grad-CAM** targets the last `Conv2D` layer, found automatically by recursing into nested models (MobileNetV2's convolutions live inside it's base), no hardcoded layer names.
- Models load **once** via a FastAPI lifespan handler, not per request.

---

## Project layout

```
dermascan/
├── backend/      FastAPI app, models, services, routers, schemas, train_and_save.py
├── frontend/     6 static pages + shared style.css + JS (vanilla, Chart.js via cdnjs)
├── docker/       Dockerfile, docker-compose.yml, nginx.conf
├── tests/        pytest (health + predict + compare)
└── README.md
```

---

## Setup

### 1. Clone & install

```bash
git clone <your-repo-url>
cd dermascan/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the models (produces the weights the API loads)

The dataset lives at the repo root in `dataset/{train,validation}/{high,low}/`.

```bash
cd backend
python train_and_save.py
# writes models/weights/cnn.keras and models/weights/mobilenet.keras
```

Until weights exist the API starts in a **degraded** state (`/health` reports it) and prediction endpoints return `500`.

### 3. Run the API

```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
# interactive docs at http://localhost:8000/docs
```

Open `frontend/index.html` (or serve the `frontend/` folder) and upload an image.

### Run with Docker

```bash
# from the dermascan/ root — train the models first so weights are baked in
docker compose -f docker/docker-compose.yml up --build
#   backend  -> http://localhost:8000
#   frontend -> http://localhost:3000   (nginx proxies /api/* to the backend)
```

The dataset is **mounted read-only** as a volume — not copied into any image.

---

## API reference

Base URL: `http://localhost:8000/api/v1`

| Method | Path | Description |
|---|---|---|
| `GET`  | `/health` | Model load status, version, uptime (seconds). |
| `POST` | `/predict` | Classify one image. Form fields: `image` (file), `model` (`both`\|`cnn`\|`mobilenet`, default `both`), `gradcam` (bool, default `true`). |
| `POST` | `/predict/compare` | Two images (`image_a`, `image_b`); returns both predictions plus a `severity_delta` saying which is worse and by how much. |

**Validation:** JPEG/PNG only, ≤ 10 MB. `400` bad/missing/oversize file or invalid `model`; `422` corrupt image; `500` model not loaded / inference failure.

Example `/predict` response:

```json
{
  "status": "success",
  "predictions": {
    "cnn":       { "label": "high", "confidence": 0.79, "raw_score": 0.21 },
    "mobilenet": { "label": "high", "confidence": 0.84, "raw_score": 0.16 }
  },
  "agreement": true,
  "gradcam_base64": "iVBORw0KGgo...",
  "gradcam_heatmap_base64": "iVBORw0KGgo...",
  "processing_time_ms": 1204
}
```

> **Label note:** `image_dataset_from_directory` orders folders alphabetically, so `high → 0`, `low → 1`. A raw sigmoid score **> 0.5 means `low`**, **≤ 0.5 means `high`**. `confidence` is the probability of the chosen class; `raw_score` is the unmodified sigmoid output.

---

## Tests

```bash
cd dermascan
pip install -r backend/requirements.txt   # includes pytest + httpx
pytest tests/
```

Covers: `/health` schema; `/predict` with a valid JPEG; `/predict` with no file (`400`) and a non-image (`400`); `model=cnn` omitting the `mobilenet` key; `/predict/compare` returning both predictions and a delta; and that both models load at startup. Inference-dependent tests **skip** (don't fail) when weights haven't been trained yet.

---

## What I learned

**Why MobileNetV2 outperformed the custom CNN.** The custom CNN learns everything from ~159 training images — far too few to discover robust visual features, so it plateaus and overfits. MobileNetV2 arrives pre-trained on ImageNet's 1.2M images: its frozen convolutional base already encodes edges, textures and colour gradients, so the only thing left to learn is a small head mapping those features to high/low severity (~164K trainable params vs millions). On this dataset that transfer-learned prior translated into noticeably higher, more stable validation accuracy.

**What the grayscale experiment revealed about colour as a feature.** Retraining both models on grayscale-converted images (architecture unchanged, colour removed) dropped accuracy for both. That tells us the models genuinely rely on colour cues — most plausibly erythema (redness) and inflammation tone, which are clinically meaningful severity signals — rather than learning texture/structure alone. Colour is a real feature here, not noise.

**How Grad-CAM confirmed the model was attending to clinically relevant regions.** Overlaying the last-conv-layer Grad-CAM heatmaps showed the activation concentrating on the affected skin lesions — inflamed, scaly or excoriated patches — rather than background, clothing or image borders. That alignment between where the model "looks" and where a clinician would look is reassuring evidence the prediction is driven by the lesion, not an artefact.

**Honest limitations.**
- **Tiny dataset** (~199 images total) — results are indicative, not robust; prone to overfitting and high variance.
- **Binary only** — "high"/"low" is a coarse simplification of a continuous clinical spectrum (e.g. EASI/SCORAD).
- **No demographic balancing** — performance across skin tones is unverified.
- **Not validated for clinical use** — no regulatory testing, no ground-truth dermatologist labelling at scale. Do not use for diagnosis.

---

## Privacy / GDPR

Uploaded images are processed **entirely in memory and never written to disk**. No images, predictions, or personal data are retained or logged after a request completes. Because nothing is stored, there is no personal data to access, export, or erase.

---

## Live demo

_Add your deployment URL here, e.g._ `https://dermascan.example.com`
