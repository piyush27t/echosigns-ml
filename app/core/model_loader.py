import os
import json
import tensorflow as tf
from typing import Dict, Any

# ------------------------------------------------------------------ #
# Global model references — loaded once at server startup              #
# Architecture: MediaPipe landmarks (42 features) → LSTM              #
# Input shape: (batch, 30, 42)                                         #
# ------------------------------------------------------------------ #
lstm_model = None
label_map: Dict[int, str] = {}

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR     = os.path.join(BASE_DIR, "models")
LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model", "lstm_model.keras")
LABELS_PATH    = os.path.join(MODELS_DIR, "labels.json")


def _configure_tensorflow():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[ML] GPU enabled: {len(gpus)} device(s)")
        except RuntimeError as e:
            print(f"[ML] GPU config error: {e}")
    else:
        print("[ML] No GPU found, running on CPU")


def _load_labels():
    global label_map
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file not found: {LABELS_PATH}")
    with open(LABELS_PATH, "r") as f:
        raw = json.load(f)
    label_map = {int(k): v for k, v in raw.items()}
    if not label_map:
        raise ValueError("Label map is empty")
    print(f"[ML] Loaded {len(label_map)} labels")


def load_models():
    """
    Called once at server startup (from run.py).
    Loads LSTM model and label map.
    Input shape: (batch, 30, 42) — 30 frames × 42 landmarks coords
    """
    global lstm_model

    _configure_tensorflow()

    if not os.path.exists(LSTM_MODEL_PATH):
        raise FileNotFoundError(f"LSTM model not found at {LSTM_MODEL_PATH}")

    print("[ML] Loading LSTM model...")
    lstm_model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
    print(f"[ML] LSTM input shape: {lstm_model.input_shape}")

    _load_labels()
    print("[ML] All models loaded successfully")


def get_models() -> Dict[str, Any]:
    """Safe accessor for predictor.py."""
    if lstm_model is None:
        raise RuntimeError("Models not loaded. Did run.py startup fail?")
    return {
        "lstm":   lstm_model,
        "labels": label_map
    }
