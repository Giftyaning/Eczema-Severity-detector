"""Train both models on the eczema dataset and save their weights.

Run this once before serving the API:

    cd dermascan/backend
    python train_and_save.py

It writes:
    models/weights/cnn.keras
    models/weights/mobilenet.keras

These are the artifacts the FastAPI app loads at startup. Training logic
mirrors the original cnn_model.py / mobilenet_model.py exactly (same
architecture, preprocessing, 10 epochs) so results match the report.
"""

from __future__ import annotations

import tensorflow as tf

from config import (
    BATCH_SIZE,
    CNN_WEIGHTS_PATH,
    EPOCHS,
    IMG_HEIGHT,
    IMG_WIDTH,
    MOBILENET_WEIGHTS_PATH,
    MODEL_DIR,
    TRAIN_DIR,
    VALIDATION_DIR,
)
from models import build_cnn, build_mobilenet


def _load_datasets():
    """Load train/validation datasets with the canonical class ordering."""
    train = tf.keras.utils.image_dataset_from_directory(
        str(TRAIN_DIR),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val = tf.keras.utils.image_dataset_from_directory(
        str(VALIDATION_DIR),
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    print("Class order (index -> name):", list(enumerate(train.class_names)))
    return train, val


def train_cnn(train, val):
    """Train the custom CNN (pixels normalised to [0, 1])."""
    print("\n=== Training custom CNN ===")
    norm = tf.keras.layers.Rescaling(1.0 / 255)
    t = train.map(lambda x, y: (norm(x), y))
    v = val.map(lambda x, y: (norm(x), y))

    model = build_cnn()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(t, validation_data=v, epochs=EPOCHS)

    model.save(CNN_WEIGHTS_PATH)
    print(f"Saved CNN -> {CNN_WEIGHTS_PATH}")


def train_mobilenet(train, val):
    """Train MobileNetV2 (its own preprocess_input -> [-1, 1])."""
    print("\n=== Training MobileNetV2 ===")
    pp = tf.keras.applications.mobilenet_v2.preprocess_input
    t = train.map(lambda x, y: (pp(x), y))
    v = val.map(lambda x, y: (pp(x), y))

    model = build_mobilenet()  # already compiled
    model.fit(t, validation_data=v, epochs=EPOCHS)

    model.save(MOBILENET_WEIGHTS_PATH)
    print(f"Saved MobileNetV2 -> {MOBILENET_WEIGHTS_PATH}")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    train, val = _load_datasets()
    train_cnn(train, val)
    train_mobilenet(train, val)
    print("\nDone. Both models saved to", MODEL_DIR)


if __name__ == "__main__":
    main()
