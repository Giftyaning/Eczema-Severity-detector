"""Grad-CAM heatmap generation.

Grad-CAM highlights the pixels that most influenced the model's decision by
weighting the final convolutional feature maps with the gradient of the
predicted score. The target convolutional layer is found automatically — never
hardcoded — so the same code works for the flat custom CNN and for MobileNetV2
(whose convolutions live inside a nested base model).
"""

from __future__ import annotations

import base64
import io

import numpy as np

from config import GRADCAM_COLORMAP, GRADCAM_OVERLAY_ALPHA, IMG_HEIGHT, IMG_WIDTH

_EPS = 1e-8


# ─────────────────────────────────────────────────────────────
# Layer discovery utilities (no hardcoded layer names)
# ─────────────────────────────────────────────────────────────


def find_last_conv_layer(model):
    """Return the last Conv2D layer in ``model``, recursing into nested models.

    Returns None if the model contains no Conv2D layer. For MobileNetV2 the
    convolutions are inside a nested functional base; this walks into it.
    """
    import tensorflow as tf

    last = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last = layer
        elif hasattr(layer, "layers"):  # nested model (e.g. MobileNetV2 base)
            nested = find_last_conv_layer(layer)
            if nested is not None:
                last = nested
    return last


def _layer_contains(container, target) -> bool:
    """True if ``target`` is ``container`` or lives somewhere inside it."""
    if container is target:
        return True
    if hasattr(container, "layers"):
        return any(_layer_contains(child, target) for child in container.layers)
    return False


def _build_grad_model(model, last_conv):
    """Build a model mapping inputs -> (last_conv activations, final output).

    Reconstructs the forward pass functionally so it works whether the target
    conv is a direct layer (CNN) or nested inside a submodel (MobileNetV2).
    """
    import tensorflow as tf

    # A fresh Input of the known shape. Relying on model.input fails in Keras 3
    # when a Sequential model hasn't been explicitly called; reconstructing the
    # forward pass from a new Input works regardless of build state.
    inp = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = inp
    conv_output = None

    for layer in model.layers:
        if conv_output is None and _layer_contains(layer, last_conv):
            if layer is last_conv:
                # Flat case (custom CNN): the conv is a direct layer.
                x = layer(x)
                conv_output = x
            else:
                # Nested case (MobileNetV2 base): split the submodel into
                # input -> conv and conv -> submodel-output, so the conv
                # activation stays on the gradient path to the final output.
                inner = layer
                sub_to_conv = tf.keras.models.Model(inner.input, last_conv.output)
                conv_output = sub_to_conv(x)
                inner_rest = tf.keras.models.Model(last_conv.output, inner.output)
                x = inner_rest(conv_output)
        else:
            x = layer(x)

    return tf.keras.models.Model(inp, [conv_output, x])


# ─────────────────────────────────────────────────────────────
# Heatmap computation
# ─────────────────────────────────────────────────────────────


def compute_heatmap(model, preprocessed_batch) -> np.ndarray:
    """Compute a normalised (0-1) Grad-CAM heatmap for a single image.

    ``preprocessed_batch`` must already be preprocessed for ``model`` and have
    shape (1, H, W, 3).
    """
    import tensorflow as tf

    last_conv = find_last_conv_layer(model)
    if last_conv is None:
        raise ValueError("Model has no Conv2D layer; cannot compute Grad-CAM.")

    grad_model = _build_grad_model(model, last_conv)

    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(preprocessed_batch)
        # Single sigmoid output: the score for the positive channel.
        score = preds[:, 0]

    grads = tape.gradient(score, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # importance per channel

    conv_out = conv_out[0]  # drop batch dim -> (h, w, channels)
    heatmap = tf.reduce_sum(conv_out * pooled_grads, axis=-1)  # weighted sum
    heatmap = tf.maximum(heatmap, 0)  # ReLU: keep positive contributions
    heatmap = heatmap / (tf.reduce_max(heatmap) + _EPS)  # normalise to 0-1
    return heatmap.numpy()


# ─────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────


def _colorize(heatmap: np.ndarray) -> np.ndarray:
    """Map a 0-1 heatmap to an RGB uint8 image using the jet colormap."""
    from matplotlib import cm

    colormap = cm.get_cmap(GRADCAM_COLORMAP)
    colored = colormap(heatmap)[..., :3]  # drop alpha
    return (colored * 255).astype("uint8")


def _to_png_base64(rgb_array: np.ndarray) -> str:
    """Encode an (H, W, 3) uint8 RGB array as a base64 PNG string."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(rgb_array).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def render(original_rgb: np.ndarray, heatmap: np.ndarray) -> tuple[str, str]:
    """Render a heatmap and a blended overlay, both as base64 PNG strings.

    ``original_rgb`` is the (H, W, 3) uint8 image the user uploaded (resized to
    the model input size). Returns (heatmap_base64, overlay_base64).
    """
    from PIL import Image

    # Resize heatmap up to the image size.
    heatmap_img = Image.fromarray((heatmap * 255).astype("uint8")).resize(
        (IMG_WIDTH, IMG_HEIGHT), Image.BILINEAR
    )
    heatmap_resized = np.asarray(heatmap_img).astype("float32") / 255.0

    colored = _colorize(heatmap_resized).astype("float32")
    original = original_rgb.astype("float32")

    # Blend: heatmap at the configured opacity over the original.
    overlay = GRADCAM_OVERLAY_ALPHA * colored + (1 - GRADCAM_OVERLAY_ALPHA) * original
    overlay = np.clip(overlay, 0, 255).astype("uint8")

    return _to_png_base64(colored.astype("uint8")), _to_png_base64(overlay)


def generate(model, model_wrapper, original_rgb: np.ndarray) -> tuple[str, str]:
    """Compute Grad-CAM for ``model`` and return (heatmap_b64, overlay_b64).

    ``model_wrapper`` provides the model-specific ``preprocess``. ``original_rgb``
    is the uploaded image resized to (H, W, 3) uint8.
    """
    batch = model_wrapper.preprocess(original_rgb)
    heatmap = compute_heatmap(model, batch)
    return render(original_rgb, heatmap)
