from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.inference import router as inference_router
from app.api.health import router as health_router
from app.core.model_loader import load_models

app = FastAPI(
    title="Sign Language Recognition ML Service",
    version="1.0.0",
    description="CNN + LSTM based real-time sign language inference service"
)

# Allow Spring Boot / frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    """
    Load ML models once at service startup.
    This avoids reloading models per request.
    """
    load_models()

# API routes
app.include_router(health_router, prefix="/health", tags=["Health"])
app.include_router(inference_router, prefix="/predict", tags=["Inference"])
