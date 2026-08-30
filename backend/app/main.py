"""
PlantCare — AI Plant Disease Detection & Care Platform
FastAPI Application Entrypoint
"""

import os
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.api.routes import router as api_router
from app.services.model_registry import model_registry
from app.services.disease_service import disease_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Pre-load default model and disease database
    print(f"Starting {settings.PROJECT_NAME} (v{settings.VERSION})...")
    disease_service.load_data()
    try:
        model_registry.get_model()  # Pre-warm default model
        print("Default model successfully warmed up.")
    except Exception as e:
        print(f"Note: Model pre-warming deferred: {e}")
    yield
    # Shutdown
    print(f"Shutting down {settings.PROJECT_NAME}...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Full-stack AI Plant Disease Detection & Care Guidance API with Grad-CAM and Explainable AI.",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows Vercel frontend, localhost, and custom domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
static_dir = Path(__file__).resolve().parent.parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
(static_dir / "examples").mkdir(parents=True, exist_ok=True)

# Mount static files for example leaf images
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Include API routes
app.include_router(api_router, prefix="/api")
app.include_router(api_router, prefix="")  # Allow root level /health as well

@app.get("/")
def root():
    return {
        "app": settings.PROJECT_NAME,
        "status": "online",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
