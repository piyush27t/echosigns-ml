import numpy as np
import cv2
from collections import deque
from typing import Tuple, Optional, Dict
import time

from app.core.model_loader import get_models
from app.core.session_manager import SessionManager
from app.preprocessing.hand_detection import extract_landmarks, FEATURES_PER_FRAME

# ------------------------------------------------------------------ #
# CONFIG — aligned with alphabet finger-spelling requirements         #
# ------------------------------------------------------------------ #
SEQUENCE_LENGTH      = 30    # Target sequence length
CONFIDENCE_THRESHOLD = 0.75  # Set to 0.75 for stricter validation
SMOOTHING_WINDOW     = 5     # Req 5+ frames of consistency

# Session manager — buffers 30 landmark vectors per user
session_manager = SessionManager(sequence_length=SEQUENCE_LENGTH)

# Dictionary to store temporal smoothing history for each user
_smoothers: Dict[str, deque] = {}

# Track the last time a stable prediction was made for UI timeouts
_last_stable_time: Dict[str, float] = {}

# Cache the built sentence
_user_sentences: Dict[str, list] = {}

# Track what sign is currently stable to prevent echoing
_currently_stable_sign: Dict[str, str] = {}

# Keep track of the hand center to detect fast motions (sign transitions)
_last_hand_center: Dict[str, Tuple[float, float]] = {}

# Track when the hand was last seen to handle buffer resets
_last_seen_hand: Dict[str, float] = {}


def _get_smoother(user_id: str) -> deque:
    if user_id not in _smoothers:
        _smoothers[user_id] = deque(maxlen=SMOOTHING_WINDOW)
    return _smoothers[user_id]


def _is_stable(smoother: deque, label: str, req_window: int) -> bool:
    """
    Return True only if the last `req_window` predictions are
    all the same label (temporal smoothing to eliminate flickering).
    """
    if len(smoother) < req_window:
        return False
    recent = list(smoother)[-req_window:]
    return all(s == label for s in recent)


def _decode_prediction(output: np.ndarray, labels: dict) -> Tuple[str, float]:
    """Convert softmax output to (label, confidence)."""
    class_id  = int(np.argmax(output))
    confidence = float(np.max(output))
    # Support both int and string keys from JSON
    text = labels.get(str(class_id), labels.get(class_id, ""))
    return text, confidence


# ------------------------------------------------------------------ #
# Main inference pipeline                                             #
# ------------------------------------------------------------------ #
def predict(user_id: str, frame: np.ndarray, timestamp: float) -> Tuple[str, float, bool, float]:
    """
    MediaPipe → Landmark extraction → LSTM inference → Temporal smoothing.

    Pipeline (per frame):
      1. MediaPipe extracts 21 hand landmarks → 42 float features
      2. Check hand motion to auto-reset sequences
      3. Features buffered in SessionManager (30-frame window)
      4. When window full → LSTM predicts label + confidence
      5. Temporal smoother checks consistency (5+ frames)
      6. Emit stable prediction with label and confidence
    """
    models   = get_models()
    lstm     = models["lstm"]
    labels   = models["labels"]
    smoother = _get_smoother(user_id)

    # Helper: Return the last known good state if we shouldn't emit a new one yet
    def get_current_ui_state():
        last_time = _last_stable_time.get(user_id, 0)
        
        # If we have no active caption, just return empty string immediately
        if last_time == 0:
            return "", 0.0, True, timestamp
            
        # Timeout after 3.0s of no stable sign to clear the sentence
        if time.time() - last_time > 3.0:
            if user_id in _last_stable_time:
                print(f"[UI] Clearing SENTENCE for {user_id}", flush=True)
            _last_stable_time.pop(user_id, None)
            _user_sentences.pop(user_id, None)
            _currently_stable_sign.pop(user_id, None)
            return "", 0.0, True, timestamp # True to ensure UI actually clears
            
        current_sentence = " ".join(_user_sentences.get(user_id, []))
        return current_sentence, 0.0, True, timestamp

    # 1. Extract landmarks (42 features)
    landmarks = extract_landmarks(frame)
    if landmarks is None:
        if user_id not in _last_seen_hand or time.time() - _last_seen_hand.get(user_id, 0) > 2.0:
             print(f"[Hand] {user_id}: No hand detected.", flush=True)
             _last_seen_hand[user_id] = time.time()
        
        smoother.clear()
        _last_hand_center.pop(user_id, None)
        return get_current_ui_state()
    else:
        if user_id not in _last_seen_hand or time.time() - _last_seen_hand.get(user_id, 0) > 5.0:
            print(f"[Hand] {user_id}: Hand DETECTED.", flush=True)
            _last_seen_hand[user_id] = time.time()

    # 2. Add Motion Reset: if hand jumps, wipe the old buffer
    # Landmarks are flattened (x,y), so we reshape to (21, 2)
    lm_arr = landmarks.reshape(-1, 2)
    cx, cy = np.mean(lm_arr[:, 0]), np.mean(lm_arr[:, 1])
    
    if user_id in _last_hand_center:
        lx, ly = _last_hand_center[user_id]
        dist = np.sqrt((cx - lx)**2 + (cy - ly)**2)
        if dist > 0.20: # 20% screen movement
            session_manager.clear_user_session(user_id)
            smoother.clear()

    _last_hand_center[user_id] = (cx, cy)

    # 3. Add to per-user frame buffer (SessionManager now handles smoothing)
    ready = session_manager.add_user_feature(user_id, landmarks)
    
    if not ready:
        return get_current_ui_state()

    # 4. LSTM inference — input shape: (1, 30, 42)
    sequence    = session_manager.get_user_sequence(user_id)
    
    # [DUAL INFERENCE] Check normal and mirrored versions to handle webcam flips
    def mirror_sequence(seq):
        """Flip x-coordinates for single hand."""
        flipped = seq.copy()
        # Negate relative x-coordinates within the hand buffer
        flipped[:, :, 0:42:2] = -flipped[:, :, 0:42:2]
        return flipped

    out_orig = lstm.predict(sequence, verbose=0)
    out_mirr = lstm.predict(mirror_sequence(sequence), verbose=0)
    
    # Pick the one with the higher maximum confidence
    conf_orig = np.max(out_orig)
    conf_mirr = np.max(out_mirr)
    lstm_output = out_mirr if conf_mirr > conf_orig else out_orig

    # 5. Result Extraction
    class_id = int(np.argmax(lstm_output[0]))
    text = labels.get(class_id, labels.get(str(class_id), "???"))
    confidence = float(np.max(lstm_output[0]))
    
    # [DIAGNOSTIC] Log Top 5 predictions
    top_indices = np.argsort(lstm_output[0])[-5:][::-1]
    top_parts = [f"{labels.get(int(idx), '???')}:{lstm_output[0][idx]:.2f}" for idx in top_indices]
    
    print(f"[Predict] {text} ({confidence:.2f}) | Top-5: {', '.join(top_parts)}", flush=True)

    # 6. Temporal smoothing
    smoother.append(text)
    
    # STABILITY THRESHOLDS
    is_stable = confidence >= CONFIDENCE_THRESHOLD and _is_stable(smoother, text, SMOOTHING_WINDOW)

    if is_stable:
        return text, confidence, True, timestamp
    else:
        return "", 0.0, False, timestamp
