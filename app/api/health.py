from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def health_check():
    """
    Health check endpoint.
    Used by backend and deployment tools.
    """
    return {
        "status": "UP",
        "service": "Sign Language Recognition ML Service"
    }
