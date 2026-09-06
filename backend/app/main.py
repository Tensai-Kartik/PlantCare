"""
PlantCare — AI Plant Disease Detection & Care Platform
FastAPI Application Entrypoint
"""

import os
import traceback
from pathlib import Path
from contextlib import asynccontextmanager
import torch

# Optimize CPU threading and memory overhead for cloud containers
torch.set_num_threads(1)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

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

# CORS configuration - Allow all origins unconditionally for seamless Vercel cross-origin communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=r".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler ensures CORS headers are always returned even on unhandled server errors
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[Server Error] Unhandled exception on {request.method} {request.url}: {exc}")
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server encountered an error: {str(exc)}"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*"
        }
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

@app.api_route("/", methods=["GET", "HEAD"])
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
