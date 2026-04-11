from fastapi import APIRouter, HTTPException

from app.schemas.request import PredictionRequest
from app.schemas.response import PredictionResponse
from app.preprocessing.frame_extractor import decode_base64_frame, FrameExtractionError
from app.core.predictor import predict

router = APIRouter()


@router.post("/", response_model=PredictionResponse)
def run_inference(request: PredictionRequest):
    """
    Endpoint called by Spring Boot backend.
    One call = one video frame.
    """

    try:
        # 1. Decode frame
        frame = decode_base64_frame(request.frame)

        # 2. Run prediction pipeline
        text, confidence, is_stable = predict(
            user_id=request.userId,
            frame=frame
        )

        # 3. Return response
        return PredictionResponse(
            text=text,
            confidence=confidence,
            is_stable=is_stable
        )

    except FrameExtractionError as e:
        # Bad frame from backend (retryable)
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        # Unexpected ML error
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
