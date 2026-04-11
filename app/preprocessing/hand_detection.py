import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import Optional

# ------------------------------------------------------------------ #
# MediaPipe Hands (Tasks API) — initialized ONCE
# ------------------------------------------------------------------ #
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "app", "models", "hand_landmarker.task")

def create_detector():
    """Attempt to initialize detector with GPU first, then fallback to CPU."""
    import time
    from mediapipe.tasks.python import BaseOptions
    
    start_time = time.time()
    try:
        print("[MediaPipe] Initializing GPU delegate...", flush=True)
        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=python.BaseOptions.Delegate.GPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=1,
            min_hand_detection_confidence=0.3,  
            min_hand_presence_confidence=0.3,   
            min_tracking_confidence=0.3         
        )
        detector = vision.HandLandmarker.create_from_options(options)
        print(f"[MediaPipe] GPU delegate initialized in {time.time() - start_time:.2f}s", flush=True)
        return detector
    except Exception as e:
        print(f"[MediaPipe] GPU delegate failed: {e}. Falling back to CPU.", flush=True)
        cpu_start = time.time()
        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH,
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options, 
            num_hands=2,
            min_hand_detection_confidence=0.3,  
            min_hand_presence_confidence=0.3,   
            min_tracking_confidence=0.3         
        )
        detector = vision.HandLandmarker.create_from_options(options)
        print(f"[MediaPipe] CPU fallback initialized in {time.time() - cpu_start:.2f}s", flush=True)
        return detector

detector = create_detector()

NUM_LANDMARKS      = 21
FEATURES_PER_FRAME = 42 # 21 landmarks * 2 coords (x, y) = 42
SEQUENCE_LENGTH    = 30 # Increased for better stability

def extract_landmarks(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Run MediaPipe Tasks on a BGR frame and return 42 features for the first hand.
    Resize frame if too large to speed up processing.
    """
    # Performance Optimization: Downscale if image is high-res
    h, w = frame.shape[:2]
    max_w = 480
    if w > max_w:
        scale = max_w / float(w)
        frame = cv2.resize(frame, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    detection_result = detector.detect(mp_image)
    if not detection_result.hand_landmarks:
        return None

    # Reverting to single-hand: Just take the first one
    landmarks = detection_result.hand_landmarks[0]
    
    # Normalize relative to the hand's center for geometric invariance
    max_lm_x = max([lm.x for lm in landmarks])
    min_lm_x = min([lm.x for lm in landmarks])
    max_lm_y = max([lm.y for lm in landmarks])
    min_lm_y = min([lm.y for lm in landmarks])
    
    mid_x = (max_lm_x + min_lm_x) / 2
    mid_y = (max_lm_y + min_lm_y) / 2
    scale = max(max_lm_x - min_lm_x, max_lm_y - min_lm_y, 1e-6)

    h_coords = np.array([[(lm.x - mid_x)/scale, (lm.y - mid_y)/scale] for lm in landmarks], dtype=np.float32).flatten()
    
    return h_coords

# ------------------------------------------------------------------ #
# Legacy helper kept for backward compatibility (not used in pipeline) #
# ------------------------------------------------------------------ #
def extract_hand_roi(frame: np.ndarray) -> Optional[np.ndarray]:
    """
    Detect hand and return a cropped ROI (BGR).
    Not used in the landmark-based LSTM pipeline.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = _hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    h, w, _ = frame.shape
    landmarks = results.multi_hand_landmarks[0].landmark

    x_coords = [int(lm.x * w) for lm in landmarks]
    y_coords = [int(lm.y * h) for lm in landmarks]

    x_min, x_max = max(min(x_coords) - 20, 0), min(max(x_coords) + 20, w)
    y_min, y_max = max(min(y_coords) - 20, 0), min(max(y_coords) + 20, h)

    if x_max <= x_min or y_max <= y_min:
        return None

    roi = frame[y_min:y_max, x_min:x_max]
    return roi if roi.size > 0 else None
