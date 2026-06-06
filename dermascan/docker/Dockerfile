# DermaScan backend image.
# Build context is the repo root (dermascan/), so paths below are relative to it:
#   docker build -f docker/Dockerfile -t dermascan-backend .
FROM python:3.10-slim

# System libs TensorFlow / Pillow need at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps first for better layer caching.
COPY backend/requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# App code.
COPY backend/ ./

# Trained weights are copied into the image (per spec). Run train_and_save.py
# locally first so backend/models/weights/*.keras exist before building.
# (.keras files in backend/models/weights/ are included by the COPY above.)

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
