from collections import deque
from typing import Dict, Deque, List
import numpy as np
import time


class UserSession:
    """
    Maintains temporal feature buffer for a single user.
    """

    def __init__(self, sequence_length: int, idle_timeout: int = 120):
        self.sequence_length = sequence_length
        self.buffer: Deque[np.ndarray] = deque(maxlen=sequence_length)
        self.last_updated = time.time()
        self.idle_timeout = idle_timeout  # seconds
        self.smoothed_feature = None

    def add_feature(self, feature: np.ndarray):
        # Apply temporal smoothing (EMA) to landmarks to reduce jitter
        if self.smoothed_feature is None:
            self.smoothed_feature = feature
        else:
            alpha = 0.4 # Smoothing factor
            self.smoothed_feature = (alpha * feature) + ((1 - alpha) * self.smoothed_feature)
            
        self.buffer.append(self.smoothed_feature)
        self.last_updated = time.time()

    def is_ready(self) -> bool:
        """
        Returns True when enough frames are collected for LSTM.
        """
        return len(self.buffer) == self.sequence_length

    def get_sequence(self) -> np.ndarray:
        """
        Returns sequence in shape:
        (1, sequence_length, feature_dim)
        """
        return np.expand_dims(np.array(self.buffer), axis=0)

    def is_idle(self) -> bool:
        """
        Check if user session is inactive.
        """
        return (time.time() - self.last_updated) > self.idle_timeout


class SessionManager:
    """
    Manages sessions for all active users.
    """

    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        self.sessions: Dict[str, UserSession] = {}

    def get_or_create_session(self, user_id: str) -> UserSession:
        if user_id not in self.sessions:
            self.sessions[user_id] = UserSession(
                sequence_length=self.sequence_length
            )
        return self.sessions[user_id]

    def cleanup_idle_sessions(self):
        """
        Remove inactive users to prevent memory leaks.
        """
        idle_users = [
            user_id for user_id, session in self.sessions.items()
            if session.is_idle()
        ]
        for user_id in idle_users:
            del self.sessions[user_id]

    def add_user_feature(self, user_id: str, feature: np.ndarray) -> bool:
        """
        Add CNN feature for a user.
        Returns True if LSTM sequence is ready.
        """
        session = self.get_or_create_session(user_id)
        session.add_feature(feature)
        self.cleanup_idle_sessions()
        return session.is_ready()

    def get_user_sequence(self, user_id: str) -> np.ndarray:
        session = self.sessions.get(user_id)
        if session is None or not session.is_ready():
            raise ValueError("Sequence not ready for user")
        return session.get_sequence()

    def clear_user_session(self, user_id: str):
        """
        Force-clear a user's frame buffer (e.g. if hand is lost).
        """
        if user_id in self.sessions:
            self.sessions[user_id].buffer.clear()
            print(f"[Session] Cleared buffer for {user_id}", flush=True)
