# Eczema-Severity-detector

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

# DermaScan

Binary eczema severity classification — **high** vs **low** — powered by two 
models running in parallel, explained by Grad-CAM heatmaps, and served through 
a FastAPI backend with a dark-themed frontend.

> ⚠️ **Not a medical device.** Educational and research project only. 
> Not validated for clinical use. Always consult a qualified dermatologist.

🔗 **Live demo:** _coming soon_

---

## What it does

Upload a skin image. DermaScan runs it through two models simultaneously — a 
custom CNN trained from scratch and a MobileNetV2 fine-tuned with transfer 
learning — and returns:

- A **high or low severity classification** with confidence score from each model
- An **agreement indicator** showing whether both models reached the same verdict
- A **Grad-CAM heatmap** overlaid on the original image showing which skin 
  regions drove the decision
- A **raw JSON response** via REST API for developers who want to integrate 
  predictions into their own applications

---

## Architecture

┌─────────────────────────────────────────────┐
Browser          │              Backend (FastAPI)               │
┌──────────┐       │  routers/predict.py  ── validate type/size  │
│ frontend │ ────► │        │             (in memory, never disk) │
│  static  │ ◄─── │        ▼                                     │
└──────────┘ JSON  │  services/predictor.py                       │
:3000           │        │  ┌── models/cnn.py       (÷255)    │
│        ├─►│   models/mobilenet.py (preproc) │
│        │  └── loaded ONCE at startup        │
│        ▼                                     │
│  services/gradcam.py ── last Conv2D (auto)  │
│                         jet heatmap+overlay │
:8000  /api/v1

**Two models, one verdict.**
- Custom CNN — 3× Conv/Pool layers + dense head, pixels normalised ÷255
- MobileNetV2 — frozen ImageNet base + GlobalAveragePooling + dense head, 
  pixels scaled to [-1, 1] via `preprocess_input`
- Both take `224×224×3` input
- `agreement` field computed from the two labels — never hardcoded
- Grad-CAM targets the last Conv2D layer found automatically — no hardcoded 
  layer names
- Models load once via FastAPI lifespan handler, not per request

---

## Project structure
Eczema-Severity-detector/
├── original_scripts/          Original training scripts — the starting point
│   ├── cnn_model.py
│   ├── mobilenet_model.py
│   ├── dataset_analysis.py
│   └── grayscale_experiment.py
├── dataset/                   199 labelled eczema images
│   ├── train/
│   │   ├── high/              121 images
│   │   └── low/               79 images (imbalanced by 42)
│   └── validation/
│       ├── high/              22 images
│       └── low/               18 images
└── dermascan/                 The built application
├── backend/
│   ├── main.py            FastAPI app entry point
│   ├── config.py          All config values centralised
│   ├── train_and_save.py  Train both models and save weights
│   ├── models/            CNN and MobileNetV2 loaders
│   ├── services/          predictor.py and gradcam.py
│   ├── routers/           predict.py and health.py
│   └── schemas/           Pydantic response models
├── frontend/              6 pages, shared CSS, vanilla JS
├── docker/                Dockerfile and docker-compose.yml
└── tests/                 9 pytest cases

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Giftyaning/Eczema-Severity-detector.git
cd Eczema-Severity-detector/dermascan/backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Train the models

```bash
python train_and_save.py
# saves models/weights/cnn.keras and models/weights/mobilenet.keras
```

The dataset lives at the repo root in `dataset/{train,validation}/{high,low}/`.
Until weights exist the API starts in degraded state — `/health` reports it 
and prediction endpoints return `500`.

### 3. Run the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
# interactive docs at http://localhost:8000/docs
```

Open `frontend/index.html` in your browser and upload an image.

### Run with Docker

```bash
docker compose -f docker/docker-compose.yml up --build
# backend  → http://localhost:8000
# frontend → http://localhost:3000
```

---

## API reference

Base URL: `http://localhost:8000/api/v1`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Model load status, version, uptime in seconds |
| `POST` | `/predict` | Classify one image — returns predictions, confidence, Grad-CAM |
| `POST` | `/predict/compare` | Two images side by side with a severity delta field |

**Parameters for `/predict`:**
- `image` — JPEG or PNG file, max 10MB (required)
- `model` — `both`, `cnn`, or `mobilenet` (default: `both`)
- `gradcam` — boolean, include heatmap in response (default: `true`)

**Error codes:** `400` bad/missing/oversize file · `422` corrupt image · 
`500` model not loaded or inference failure

**Example response:**

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

> **Label mapping:** `image_dataset_from_directory` assigns class indices 
> alphabetically — `high → 0`, `low → 1`. A sigmoid output **> 0.5 means low**, 
> **≤ 0.5 means high**. `confidence` is the probability of the predicted class. 
> `raw_score` is the unmodified sigmoid output.

---

## Tests

```bash
cd Eczema-Severity-detector/dermascan
pytest tests/ -v
```

Covers: `/health` schema · `/predict` with valid JPEG · `/predict` with no 
file (400) · `/predict` with non-image file (400) · `model=cnn` omitting 
mobilenet key · `/predict/compare` returning delta · both models loading at 
startup. Inference tests skip gracefully when weights are absent.

---

## What I learned

**Why MobileNetV2 outperformed the custom CNN.**
The custom CNN learns everything from roughly 159 training images — far too 
few to discover robust visual features, so it overfits quickly. MobileNetV2 
arrives pre-trained on 1.2 million ImageNet images. Its frozen convolutional 
base already encodes edges, textures, and colour gradients. The only thing 
left to learn is a small classification head mapping those features to 
high/low severity — around 164K trainable parameters versus millions for the 
CNN. That transfer-learned prior translated into noticeably higher and more 
stable validation accuracy on this small dataset.

**What the grayscale experiment revealed.**
Retraining both models on grayscale images — architecture unchanged, colour 
removed — dropped accuracy for both. This confirmed that the models rely on 
colour cues, most plausibly erythema (skin redness) and inflammation tone, 
which are clinically meaningful severity signals. Colour is a genuine 
predictive feature here, not noise.

**How Grad-CAM confirmed clinical relevance.**
Overlaying Grad-CAM heatmaps on validation images showed activations 
concentrating on the inflamed, scaly, or excoriated skin patches — not 
background, clothing, or image borders. That alignment between where the 
model attends and where a clinician would look is reassuring evidence the 
prediction is driven by the lesion itself, not an artefact.

**A label mapping bug caught before deployment.**
`image_dataset_from_directory` assigns class indices alphabetically, meaning 
`high → 0` and `low → 1`. The original training scripts commented that a 
sigmoid output above 0.5 meant high severity — the opposite of the truth. 
This was caught during code analysis before the API was built and corrected 
throughout the codebase.

**Honest limitations.**
- Tiny dataset (~199 images) — results are indicative, not robust
- Binary classification only — high/low is a coarse simplification of a 
  continuous clinical spectrum such as EASI or SCORAD
- No demographic balancing — performance across skin tones is unverified
- Not validated for clinical use — no regulatory testing, no ground-truth 
  dermatologist labelling at scale

---

## Privacy and GDPR

Uploaded images are processed entirely in memory and never written to disk. 
No images, predictions, or personal data are retained after a request 
completes. Nothing is stored, so there is no personal data to access, 
export, or erase.

---

## Tech stack

| Layer | Technology |
|-------|------------|
| ML framework | TensorFlow 2.20 / Keras |
| API | FastAPI 0.111 + Uvicorn |
| Validation | Pydantic v2 |
| Image processing | Pillow, NumPy |
| Explainability | Grad-CAM via TensorFlow GradientTape |
| Frontend | Vanilla HTML, CSS, JavaScript, Chart.js |
| Containerisation | Docker + nginx |
| CI/CD | GitHub Actions |
| Deployment | Render |

---

## License

MIT