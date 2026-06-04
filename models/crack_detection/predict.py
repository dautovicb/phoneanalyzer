"""Crack detection — loads the Keras crack classifier and scores phone crops.

Framework-agnostic: the model is lazily loaded once and cached in a module-level
singleton, so it persists across Streamlit reruns without depending on Streamlit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_CRACK_MODEL_PATH = Path(__file__).resolve().parent / "model" / "crack_detector.keras"

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


def predict_crack(crop: Image.Image) -> tuple[bool, float]:
    """Return (is_cracked, confidence) for a phone crop. Sigmoid score < 0.5 => cracked."""
    import tensorflow as tf

    model = load_crack_model()
    img = crop.convert("RGB").resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    score = float(predictions.item())

    if score < 0.5:
        return True, 1 - score
    return False, score
