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


import threading
import time

# Set to track users currently in processing to skip frames and avoid lag
processing_users = set()
processing_lock = threading.Lock()

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

        # 3. Emit prediction back to the room
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
            print(f"[Performance] User {user_name}: Predicted '{text}' (conf={confidence:.2f}) in {duration:.0f}ms")
        else:
            if duration > 100: # Only log slow "No hand" detections
                print(f"[Performance] User {user_name}: No detection in {duration:.0f}ms")

    except FrameExtractionError as e:
        print(f"[ML Server] Frame decode error for {user_id}: {e}")
    except Exception as e:
        print(f"[ML Server] Inference error for {user_id}: {e}")
    finally:
        with processing_lock:
            processing_users.discard(user_id)
