from pydantic import BaseModel, Field
from typing import Optional


class PredictionRequest(BaseModel):
    """
    Request schema expected from Spring Boot backend.
    One request corresponds to ONE video frame.
    """

    userId: str = Field(
        ...,
        description="Unique user identifier (roomId_userId)"
    )

    frame: str = Field(
        ...,
        description="Base64 encoded image frame (JPEG/PNG)"
    )

    timestamp: float = Field(
        ...,
        description="Unix timestamp when frame was captured"
    )

    sessionId: Optional[str] = Field(
        None,
        description="Optional session/room identifier"
    )
