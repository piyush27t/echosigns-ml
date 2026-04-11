from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    """
    Response schema returned to Spring Boot backend.
    """

    text: str = Field(
        ...,
        description="Predicted sentence or partial caption"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence score"
    )

    is_stable: bool = Field(
        ...,
        description="Whether the prediction is stable enough to broadcast"
    )
