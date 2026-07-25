"""Crack detection — loads the Keras crack classifier and scores phone crops.

Framework-agnostic: the model is lazily loaded once and cached in a module-level
singleton, so it persists across Streamlit reruns without depending on Streamlit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

# v4 (ConvNeXt-Tiny, 384x384). Benchmarked over 54 labelled OLX listings by
# models/crack_detection/test_listings.py: P 0.944 / R 0.872 / F1 0.907 on the
# flag decision, vs v3's 0.935 / 0.744 / 0.829 — the gain is front-crack recall
# (0.286 -> 0.714), which was the weak class.
#
# This is the *_local export deliberately: crack_detector_v4.keras was saved by
# Keras 3.13, which writes `quantization_config` into every Dense config and
# cannot be read by the 3.12 runtime we pin. The _local file is the same network
# re-exported by 3.12 without optimizer state, so it loads on both.
DEFAULT_CRACK_MODEL_PATH = Path(__file__).resolve().parent / "model" / "crack_detector_v4_local.keras"

_model = None


def _load_keras_model(model_path: Path):
    """Load a .keras model, falling back to legacy deserialization on TF/Keras mismatch."""
    try:
        import tensorflow as tf

        return tf.keras.models.load_model(str(model_path), compile=False)
    except TypeError as exc:
        try:
            import keras

            if hasattr(keras, "config") and hasattr(keras.config, "enable_legacy_deserialization"):
                keras.config.enable_legacy_deserialization()

            if hasattr(keras, "saving") and hasattr(keras.saving, "load_model"):
                return keras.saving.load_model(str(model_path), compile=False)

            return keras.models.load_model(str(model_path), compile=False)
        except Exception:
            raise exc


def load_crack_model(model_path: Path = DEFAULT_CRACK_MODEL_PATH):
    """Lazily load and cache the crack-detection model."""
    global _model
    if _model is None:
        _model = _load_keras_model(model_path)
    return _model


def _model_input_size(model, default: int = 384) -> int:
    """Square input side the loaded model expects, read off the model itself.

    Deliberately not a constant: a hardcoded resize is exactly how this broke
    before — the size moved to 384 for the ConvNeXt models while the default
    weights were still the 224-input v1, and every call raised ValueError deep
    inside Keras. Reading it from the model means the two can never disagree.
    """
    shape = getattr(model, "input_shape", None)
    if isinstance(shape, list) and shape:
        shape = shape[0]
    if shape and len(shape) == 4 and shape[1] and shape[2]:
        return int(shape[1])
    return default


def predict_crack(crop: Image.Image) -> tuple[bool, float]:
    """Return (is_cracked, confidence) for a phone crop. Sigmoid score < 0.5 => cracked."""
    import tensorflow as tf

    model = load_crack_model()
    size = _model_input_size(model)
    img = crop.convert("RGB").resize((size, size))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    score = 1 - float(predictions.item())

    if score > 0.5:
        return True, score
    return False, score

