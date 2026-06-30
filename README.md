# DermaScan

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![AWS EC2](https://img.shields.io/badge/AWS-EC2-ff9900)
![License](https://img.shields.io/badge/License-MIT-yellow)

Binary eczema severity classification — **high** vs **low** — powered by two 
models running in parallel, explained by Grad-CAM heatmaps, and served through 
a FastAPI backend with a dark-themed frontend.

> ⚠️ **Not a medical device.** Educational and research project only. 
> Not validated for clinical use. Always consult a qualified dermatologist.

🔗 **Live demo:** http://dermascan.duckdns.org:8080

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

```

┌─────────────────────────────────────────────┐
Browser          │              Backend (FastAPI)               │
┌──────────┐       │  routers/predict.py  ── validate type/size  │
│ frontend │ ────► │        │             (in memory, never disk) │
│  static  │ ◄─── │        ▼                                     │
└──────────┘ JSON  │  services/predictor.py                       │
:8080           │        │  ┌── models/cnn.py       (÷255)    │
│        ├─►│   models/mobilenet.py (preproc) │
│        │  └── loaded ONCE at startup        │
│        ▼                                     │
│  services/gradcam.py ── last Conv2D (auto)  │
│                         jet heatmap+overlay │
:8000  /api/v1

```

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

```
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
```
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

## Deployment

DermaScan runs on a single **AWS EC2 t3.micro** instance (free tier) on 
**Amazon Linux 2023** with **Python 3.11**.

- **Backend** — the FastAPI app runs on port `8000` under uvicorn. To survive 
  SSH disconnects, uvicorn is launched in the background with `nohup` so the 
  process persists across sessions.
- **Frontend** — the static pages are served from the same instance on port 
  `8080` via Python's built-in HTTP server (`python -m http.server 8080`).
- **Networking** — both ports `8000` and `8080` are opened to inbound traffic 
  through AWS security group rules.
- **Model weights** — t3.micro's 1GB RAM is not enough to train the models 
  in place, so the weights were trained locally and transferred to the 
  instance via `scp`.

```bash
# on the EC2 instance, from dermascan/backend
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > uvicorn.log 2>&1 &

# from dermascan/frontend
nohup python -m http.server 8080 > frontend.log 2>&1 &
```

---

## Honest deployment challenges (what I learned)

Getting this onto a live server was where most of the real learning happened. 
The issues below are the ones that actually cost time — and how each was 
resolved.

- **Render's free tier ran out of memory.** 512MB RAM could not load both 
  TensorFlow models at once. Migrated to AWS EC2 to get the 1GB tier.
- **EC2 killed the training process.** Even on EC2, t3.micro's memory limits 
  killed training mid-run. Solved by training locally and transferring the 
  finished `.keras` weights via `scp`.
- **Keras version mismatch broke model loading.** Weights saved on my local 
  Mac (Keras 3.14) failed to deserialize on EC2, which shipped Keras 3.10 on 
  Python 3.9. Solved by installing Python 3.11 on EC2 so the Keras version 
  matched 3.14.
- **EBS default storage was too small.** The default 8GB volume could not 
  hold TensorFlow (a 590MB wheel) plus dependencies. Extended the volume to 
  20GB via an EBS volume modification, then grew the filesystem in place with 
  `growpart` and `xfs_growfs`.
- **Matplotlib deprecated an API mid-stack.** Matplotlib 3.9 removed 
  `cm.get_cmap`, which the Grad-CAM renderer relied on. Patched to use 
  `plt.get_cmap` instead.
- **A label mapping bug, caught early.** `image_dataset_from_directory` orders 
  folders alphabetically, so `high → 0` and `low → 1`. A sigmoid output 
  **> 0.5 means low**, not high. This was caught during the initial code 
  analysis and corrected throughout.

---

## API reference

Base URL: `http://54.226.38.244:8000/api/v1` (live) or `http://localhost:8000/api/v1` (local)

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
The custom CNN learns everything from ~159 training images — far too few to 
discover robust visual features, so it overfits quickly. MobileNetV2 arrives 
pre-trained on 1.2 million ImageNet images, its frozen convolutional base 
already encoding edges, textures, and colour gradients. All that is left to 
learn is a small classification head — roughly 164K trainable parameters 
versus millions for the CNN. That transfer-learned prior translated into 
noticeably higher and more stable validation accuracy on this small dataset.

**What the grayscale experiment revealed.**
Retraining both models on grayscale images — architecture unchanged, colour 
removed — dropped accuracy for both. This confirmed the models rely on colour 
cues, most plausibly erythema (skin redness) and inflammation tone, which are 
clinically meaningful severity signals. Colour is a genuine predictive feature 
here, not noise.

**How Grad-CAM confirmed clinical relevance.**
Overlaying Grad-CAM heatmaps on validation images showed activations 
concentrating on the inflamed, scaly, or excoriated skin patches — not 
background, clothing, or image borders. That alignment between where the model 
attends and where a clinician would look is reassuring evidence the prediction 
is driven by the lesion itself, not an artefact.

**A label mapping bug caught before deployment.**
`image_dataset_from_directory` assigns class indices alphabetically, so 
`high → 0` and `low → 1`. The original training scripts commented that a 
sigmoid output above 0.5 meant high severity — the opposite of the truth. 
Caught during code analysis before the API was built, and corrected throughout.

**Honest limitations.**
- Tiny dataset (~199 images) — results are indicative, not robust
- Binary classification only — high/low is a coarse simplification of a 
  continuous clinical spectrum such as EASI or SCORAD
- No demographic balancing — performance across skin tones is unverified
- Not validated for clinical use — no regulatory testing, no ground-truth 
  dermatologist labelling at scale

---

## Live demo notes

- The live demo currently uses **HTTP, not HTTPS** — acceptable for a 
  portfolio project.
- The EC2 **free tier IP may change** if the instance is stopped and 
  restarted.
- If the demo URL is unresponsive, the **EC2 instance may need to be 
  restarted**.
- For production I would add a **domain name**, an **SSL certificate via 
  Let's Encrypt**, and an **Elastic IP** for a stable address.

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
| ML framework | TensorFlow 2.16 / Keras |
| API | FastAPI 0.111 + Uvicorn |
| Validation | Pydantic v2 |
| Image processing | Pillow, NumPy |
| Explainability | Grad-CAM via TensorFlow GradientTape |
| Frontend | Vanilla HTML, CSS, JavaScript, Chart.js |
| Containerisation | Docker + nginx |
| CI/CD | GitHub Actions |
| Deployment | AWS EC2 (t3.micro, Amazon Linux 2023) |

---

## License

MIT
