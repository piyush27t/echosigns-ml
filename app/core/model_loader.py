import os
import json
import tensorflow as tf
import numpy as np
from typing import Dict, Any
import h5py
import tempfile
import shutil
import zipfile

# ------------------------------------------------------------------ #
# Global model references — loaded once at server startup              #
# Architecture: MediaPipe landmarks (42 features) → LSTM              #
# Input shape: (batch, 30, 42)                                         #
# ------------------------------------------------------------------ #
lstm_model = None
label_map: Dict[int, str] = {}

BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR     = os.path.join(BASE_DIR, "models")
# Prefer .h5 as it's more stable for metadata manipulation via h5py
KERAS_MODEL_PATH_H5 = os.path.join(MODELS_DIR, "lstm_model", "lstm_model.h5")
KERAS_MODEL_PATH_KERAS = os.path.join(MODELS_DIR, "lstm_model", "lstm_model.keras")
KERAS_MODEL_PATH = KERAS_MODEL_PATH_H5 if os.path.exists(KERAS_MODEL_PATH_H5) else KERAS_MODEL_PATH_KERAS
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
    Loads Keras LSTM model with quantization_config removed.
    Input shape: (batch, 30, 42) — 30 frames × 42 landmarks coords
    """
    global lstm_model

    _configure_tensorflow()

    if not os.path.exists(KERAS_MODEL_PATH):
        raise FileNotFoundError(f"Keras model not found at {KERAS_MODEL_PATH}")

    print("[ML] Loading Keras LSTM model...")
    try:
        # First attempt: direct load
        lstm_model = tf.keras.models.load_model(KERAS_MODEL_PATH, compile=False)
        print("[ML] Model loaded successfully")
    except TypeError as e:
        if "quantization_config" in str(e):
            print("[ML] Removing quantization_config from model file...")
            # Load model, remove quantization_config, save to temp file, then load
            def remove_quantization_from_keras(model_path):
                """Remove quantization_config from .keras (ZIP) or .h5 (HDF5) file"""
                def recursive_remove(obj):
                    if isinstance(obj, dict):
                        obj.pop('quantization_config', None)
                        for v in obj.values():
                            recursive_remove(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            recursive_remove(item)

                if zipfile.is_zipfile(model_path):
                    print(f"[ML] Detected ZIP archive (.keras format) at {model_path}")
                    # Keras 3 format is a ZIP containing config.json
                    tmp_zip = model_path + ".tmp.zip"
                    with zipfile.ZipFile(model_path, 'r') as zin:
                        with zipfile.ZipFile(tmp_zip, 'w') as zout:
                            for item in zin.infolist():
                                buffer = zin.read(item.filename)
                                if item.filename == 'config.json':
                                    config = json.loads(buffer.decode('utf-8'))
                                    recursive_remove(config)
                                    buffer = json.dumps(config).encode('utf-8')
                                zout.writestr(item, buffer)
                    shutil.move(tmp_zip, model_path)
                else:
                    print(f"[ML] Detected HDF5 format (.h5) at {model_path}")
                    with h5py.File(model_path, 'r+') as f:
                        if 'model_config' in f.attrs:
                            config_str = f.attrs['model_config'].decode('utf-8') if isinstance(f.attrs['model_config'], bytes) else f.attrs['model_config']
                            config = json.loads(config_str)
                            recursive_remove(config)
                            f.attrs['model_config'] = json.dumps(config).encode('utf-8')
                        else:
                            print("[ML] Warning: 'model_config' not found in HDF5 attributes")
            
            # Work on a temp copy to avoid corrupting original
            with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp:
                shutil.copy2(KERAS_MODEL_PATH, tmp.name)
                remove_quantization_from_keras(tmp.name)
                lstm_model = tf.keras.models.load_model(tmp.name, compile=False)
                os.unlink(tmp.name)
            print("[ML] Model loaded successfully (quantization_config removed)")
        else:
            raise
    
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
