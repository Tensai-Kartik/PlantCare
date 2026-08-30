"""
REST API Endpoints for PlantCare
Handles health check, model management, image quality assessment, full plant diagnosis,
multi-model consensus comparison, disease knowledge base, and curated examples.
"""

from typing import Optional, List
from pathlib import Path
from PIL import Image
from io import BytesIO
from fastapi import APIRouter, File, UploadFile, Query, Form, HTTPException, Request, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.rate_limiter import rate_limiter, inference_semaphore
from app.schemas.analysis import (
    AnalysisResponse,
    QualityCheckResult,
    ModelListResponse,
    ModelDisagreementResult,
    ExampleLeaf
)
from app.schemas.disease import DiseaseListResponse, DiseaseInfo
from app.services.image_quality import image_quality_service
from app.services.model_registry import model_registry
from app.services.disease_service import disease_service
from app.services.inference import inference_service

router = APIRouter()

# Curated example leaves available for instant 1-click evaluation
SAMPLE_EXAMPLES = [
    ExampleLeaf(
        id="tomato_early_blight",
        title="Tomato Early Blight",
        plant="Tomato",
        condition="Alternaria solani",
        image_url="/examples/tomato_early_blight.jpg",
        is_healthy=False
    ),
    ExampleLeaf(
        id="potato_late_blight",
        title="Potato Late Blight",
        plant="Potato",
        condition="Phytophthora infestans",
        image_url="/examples/potato_late_blight.jpg",
        is_healthy=False
    ),
    ExampleLeaf(
        id="apple_scab",
        title="Apple Scab",
        plant="Apple",
        condition="Venturia inaequalis",
        image_url="/examples/apple_scab.jpg",
        is_healthy=False
    ),
    ExampleLeaf(
        id="grape_black_rot",
        title="Grape Black Rot",
        plant="Grape",
        condition="Guignardia bidwellii",
        image_url="/examples/grape_black_rot.jpg",
        is_healthy=False
    ),
    ExampleLeaf(
        id="pepper_bell_bacterial_spot",
        title="Pepper Bacterial Spot",
        plant="Pepper",
        condition="Xanthomonas euvesicatoria",
        image_url="/examples/pepper_bell_bacterial_spot.jpg",
        is_healthy=False
    ),
    ExampleLeaf(
        id="tomato_healthy",
        title="Tomato Healthy Leaf",
        plant="Tomato",
        condition="Solanum lycopersicum",
        image_url="/examples/tomato_healthy.jpg",
        is_healthy=True
    )
]

ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
MAX_FILE_SIZE_BYTES = settings.MAX_FILE_SIZE_BYTES

@router.get("/health", summary="Health Check")
async def health_check():
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "default_model": model_registry.default_model_id,
        "calibration_enabled": settings.ENABLE_TEMPERATURE_CALIBRATION,
        "gemini_vision_mode": settings.GEMINI_VISION_MODE
    }

@router.get("/models", response_model=ModelListResponse, summary="List Available AI Models")
async def get_available_models():
    return model_registry.list_models()

@router.post("/quality-check", response_model=QualityCheckResult, summary="Assess Image Suitability & Multi-signal Presence")
async def check_image_suitability(
    request: Request,
    file: UploadFile = File(...)
):
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not await rate_limiter.check(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment before sending more requests."
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Please upload JPG, PNG, or WEBP."
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)} MB limit."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    try:
        return image_quality_service.evaluate_image(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to process image: {str(e)}"
        )

@router.post("/analyze", response_model=AnalysisResponse, summary="Perform Full Plant Pathology Analysis")
async def analyze_plant_leaf(
    request: Request,
    file: UploadFile = File(...),
    model_id: Optional[str] = Form(None),
    skip_quality_check: bool = Form(False),
    enable_model_comparison: bool = Form(False)
):
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not await rate_limiter.check(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please wait a moment before analyzing more images."
        )

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file.content_type}'. Please upload JPG, PNG, or WEBP."
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size exceeds {MAX_FILE_SIZE_BYTES // (1024*1024)} MB limit."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty."
        )

    async with inference_semaphore:
        try:
            return inference_service.run_analysis(
                image_bytes=content,
                model_id=model_id,
                skip_quality_check=skip_quality_check,
                enable_model_comparison=enable_model_comparison
            )
        except Exception as e:
            print(f"Analysis error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Analysis pipeline error: {str(e)}"
            )

@router.post("/compare-models", response_model=ModelDisagreementResult, summary="Verify Multi-Model Agreement")
async def compare_models_endpoint(
    request: Request,
    file: UploadFile = File(...)
):
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not await rate_limiter.check(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded."
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="File size exceeds limit.")

    try:
        pil_image = Image.open(BytesIO(content)).convert("RGB")
        return model_registry.run_model_comparison(pil_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")

@router.get("/examples", response_model=List[ExampleLeaf], summary="Get Curated Example Leaf Images")
async def get_examples():
    return SAMPLE_EXAMPLES

@router.post("/analyze-example/{example_id}", response_model=AnalysisResponse, summary="Analyze Built-in Example")
async def analyze_example(
    example_id: str,
    model_id: Optional[str] = Query(None),
    enable_model_comparison: bool = Query(False)
):
    # Locate sample in backend static dir or dataset
    sample_path = settings.STATIC_DIR / "examples" / f"{example_id}.jpg"
    if not sample_path.exists():
        # Fallback to dataset raw
        sample_path = settings.BASE_DIR.parent / "ml" / "dataset" / "raw" / example_id / "leaf_0001.jpg"

    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Example '{example_id}' not found.")

    with open(sample_path, "rb") as f:
        content = f.read()

    async with inference_semaphore:
        return inference_service.run_analysis(
            image_bytes=content,
            model_id=model_id,
            enable_model_comparison=enable_model_comparison
        )

@router.get("/diseases", response_model=DiseaseListResponse, summary="Search and Filter Disease Knowledge Base")
async def list_diseases(
    plant: Optional[str] = Query(None, description="Filter by crop (e.g. Tomato, Potato, Apple)"),
    severity: Optional[str] = Query(None, description="Filter by severity level"),
    q: Optional[str] = Query(None, description="Search keyword in disease name, symptoms, or description")
):
    return disease_service.list_all(plant=plant, severity=severity, query=q)

@router.get("/diseases/{disease_id}", response_model=DiseaseInfo, summary="Get Specific Disease Details")
async def get_disease_detail(disease_id: str):
    info = disease_service.get_by_id(disease_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Disease with ID '{disease_id}' not found.")
    return info
