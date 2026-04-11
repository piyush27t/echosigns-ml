"""
Entry point for the Echosigns ML Socket.IO server.

Usage:
    python run.py

The server:
- Listens on http://0.0.0.0:8080 (matches frontend expectation)
- Handles Socket.IO events: connect, disconnect, frame
- Emits: prediction
"""

import os
import sys

# Ensure ml-service root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.model_loader import load_models
from app.server import socketio, app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))

    # Load ML models ONCE before serving any requests
    print(f"[ML Server] Loading models (this may take a moment)...")
    load_models()

    print(f"[ML Server] Starting Socket.IO server on http://0.0.0.0:{port}")
    socketio.run(app, host="0.0.0.0", port=port, debug=False, allow_unsafe_werkzeug=True)
