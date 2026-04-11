import base64
import cv2
import numpy as np


class FrameExtractionError(Exception):
    """Raised when frame decoding fails."""
    pass


def decode_base64_frame(base64_str: str) -> np.ndarray:
    """
    Decode a base64-encoded image into an OpenCV BGR frame.

    Expected input:
    - Base64 string (JPEG or PNG)
    - No data URI prefix required

    Returns:
    - OpenCV image (numpy array, BGR)
    """

    try:
        # Decode base64 string to bytes
        image_bytes = base64.b64decode(base64_str)

        # Convert bytes to numpy array
        np_buffer = np.frombuffer(image_bytes, dtype=np.uint8)

        # Decode image using OpenCV
        frame = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

        if frame is None:
            raise FrameExtractionError("OpenCV failed to decode image")

        return frame

    except Exception as e:
        raise FrameExtractionError(f"Frame decoding failed: {str(e)}")
