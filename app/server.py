import os
from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room

app = Flask(__name__)
# Enable CORS for all origins (restrict in production)
# Switching to 'threading' for stability on Windows + Python 3.13
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='threading',
    logger=True, 
    engineio_logger=True
)


@socketio.on('connect')
def handle_connect():
    """
    Handle new connection.
    Frontend sends query params: roomId, userId, userName
    """
    room_id = request.args.get('roomId')
    user_id = request.args.get('userId')
    user_name = request.args.get('userName')

    if not all([room_id, user_id]):
        print(f"[ML Server] Connection rejected: Missing roomId or userId")
        return False

    join_room(room_id)
    print(f"[ML Server] User {user_name} ({user_id}) joined room {room_id}")


@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.args.get('userId')
    print(f"[ML Server] User {user_id} disconnected")


@socketio.on('reset_prediction_state')
def handle_reset_prediction_state(data):
    """
    Handle reset_prediction_state event from frontend.
    Clears all cached frames and model state for the user.
    Forces fresh detection from new frames onwards.
    Payload: { userId }
    """
    from app.core.predictor import reset_user_prediction_state
    
    user_id = data.get('userId')
    if not user_id:
        print("[ML Server] Reset request missing userId")
        return
    
    print(f"[ML Server] Reset prediction state for user {user_id}")
    reset_user_prediction_state(user_id)
    
    # Also clear debouncing state for this user
    with debounce_lock:
        last_emitted_predictions.pop(user_id, None)
    
    # Optionally emit acknowledgment back to frontend
    emit('reset_prediction_state_ack', {
        'userId': user_id,
        'status': 'success'
    })


import threading
import time

# Set to track users currently in processing to skip frames and avoid lag
processing_users = set()
processing_lock = threading.Lock()

# Debouncing: Track last emitted prediction per user
# { user_id: { 'text': str, 'timestamp': float } }
last_emitted_predictions = {}
debounce_lock = threading.Lock()
DEBOUNCE_INTERVAL = 0.5  # 500ms minimum between emissions for same prediction


def should_emit_prediction(user_id: str, text: str) -> bool:
    """
    Check if prediction should be emitted based on debouncing logic.
    
    Emit if:
    - Text is different from last emitted prediction, OR
    - At least DEBOUNCE_INTERVAL (500ms) has passed since last emission
    
    Don't emit empty/no-hand predictions consecutively.
    """
    with debounce_lock:
        current_time = time.time()
        last_pred = last_emitted_predictions.get(user_id)
        
        # First prediction for this user
        if last_pred is None:
            last_emitted_predictions[user_id] = {
                'text': text,
                'timestamp': current_time
            }
            return True
        
        last_text = last_pred['text']
        last_time = last_pred['timestamp']
        time_since_last = current_time - last_time
        
        # Text changed: emit immediately
        if text != last_text:
            last_emitted_predictions[user_id] = {
                'text': text,
                'timestamp': current_time
            }
            # Only log text changes (ignore empty/no-hand changes)
            if text and last_text:
                print(f"[Debounce] User {user_id}: Text changed from '{last_text}' to '{text}'", flush=True)
            return True
        
        # Same text: only emit if enough time has passed
        if time_since_last >= DEBOUNCE_INTERVAL:
            last_emitted_predictions[user_id]['timestamp'] = current_time
            return True
        
        # Debounced: don't emit
        return False


@socketio.on('frame')
def handle_frame(data):
    """
    Handle 'frame' event from frontend.
    Payload: { roomId, userId, userName, frame (Base64 JPEG), timestamp }
    """
    from app.preprocessing.frame_extractor import decode_base64_frame, FrameExtractionError
    from app.core.predictor import predict

    user_id = data.get('userId')
    
    # Congestion Control: Skip frame if already processing for this user
    with processing_lock:
        if user_id in processing_users:
            # print(f"[Server] Skipping frame for {user_id} (Busy)")
            return
        processing_users.add(user_id)

    start_perf = time.time()
    try:
        room_id = data.get('roomId')
        user_name = data.get('userName')
        base64_frame = data.get('frame')
        timestamp = data.get('timestamp')

        if not base64_frame:
            return

        # 1. Decode base64 JPEG → OpenCV frame
        frame = decode_base64_frame(base64_frame)

        # 2. Run ML inference pipeline
        # Returns: (text, confidence, is_stable, original_timestamp)
        text, confidence, is_stable, original_timestamp = predict(
            user_id=user_id,
            frame=frame,
            timestamp=timestamp
        )

        # 3. Debounce: Only emit if prediction is new or enough time has passed
        if should_emit_prediction(user_id, text):
            # Emit prediction back to the room
            emit('prediction', {
                'userId': user_id,
                'userName': user_name,
                'label': text,
                'text': text, # Legacy support
                'confidence': confidence,
                'is_stable': is_stable,
                'timestamp': original_timestamp
            }, room=room_id)

            duration = (time.time() - start_perf) * 1000
            if text:
                print(f"[Performance] User {user_name}: Predicted '{text}' (conf={confidence:.2f}) in {duration:.0f}ms", flush=True)
            else:
                if duration > 100: # Only log slow "No hand" detections
                    print(f"[Performance] User {user_name}: No detection in {duration:.0f}ms", flush=True)

    except FrameExtractionError as e:
        print(f"[ML Server] Frame decode error for {user_id}: {e}")
    except Exception as e:
        print(f"[ML Server] Inference error for {user_id}: {e}")
    finally:
        with processing_lock:
            processing_users.discard(user_id)
