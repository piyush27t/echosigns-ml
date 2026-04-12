import os
import json
import tensorflow as tf
import numpy as np
from typing import Dict, Any

# ------------------------------------------------------------------ #
# Global model references — loaded once at server startup              #
# Architecture: MediaPipe landmarks (42 features) → LSTM              #
# Input shape: (batch, 30, 42)                                         #
# Using TFLite for optimized inference on Render                      #
# ------------------------------------------------------------------ #
tflite_interpreter = None
label_map: Dict[int, str] = {}

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR     = os.path.join(BASE_DIR, "models")
TFLITE_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model", "model.tflite")
KERAS_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model", "lstm_model.keras")
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
    Loads TFLite model (optimized for inference) and label map.
    Input shape: (batch, 30, 42) — 30 frames × 42 landmarks coords
    TFLite provides ~5-10x faster inference than full Keras model.
    """
    global tflite_interpreter

    _configure_tensorflow()

    # Try TFLite first (preferred for performance)
    if os.path.exists(TFLITE_MODEL_PATH):
        print("[ML] Loading TFLite model (optimized)...")
        try:
            tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
            tflite_interpreter.allocate_tensors()
            print("[ML] TFLite model loaded successfully")
            print(f"[ML] Input details: {tflite_interpreter.get_input_details()}")
            print(f"[ML] Output details: {tflite_interpreter.get_output_details()}")
        except Exception as e:
            print(f"[ML] Failed to load TFLite model: {e}")
            raise
    else:
        raise FileNotFoundError(f"TFLite model not found at {TFLITE_MODEL_PATH}")

    _load_labels()
    print("[ML] All models loaded successfully")


def get_models() -> Dict[str, Any]:
    """Safe accessor for predictor.py."""
    if tflite_interpreter is None:
        raise RuntimeError("Models not loaded. Did run.py startup fail?")
    return {
        "tflite": tflite_interpreter,
        "labels": label_map
    }
