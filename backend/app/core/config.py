"""
Application Configuration and Settings for PlantCare Backend
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "PlantCare — AI Plant Health Detector"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"

    # Environment
    ENV: str = os.getenv("ENV", "development")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5173")

    # ML Model Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "efficientnet_b0")
    MODEL_WEIGHTS_DIR: Path = PROJECT_ROOT / "model_weights"
    DISEASE_DATA_PATH: Path = PROJECT_ROOT / "data" / "diseases.json"
    STATIC_DIR: Path = PROJECT_ROOT / "static"

    # LLM Settings
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

    # Multi-Model Consensus Default
    ENABLE_MULTI_MODEL_CONSENSUS: bool = True

    # Confidence & Uncertainty Thresholds
    CONFIDENCE_HIGH: float = 0.75
    CONFIDENCE_MODERATE: float = 0.45
    CONFIDENCE_LOW: float = 0.30
    UNCERTAINTY_ENTROPY_THRESHOLD: float = 1.80  # Shannon entropy threshold for uncertain diagnosis
    TOP_MARGIN_THRESHOLD: float = 0.15           # Min margin between Top-1 and Top-2 to consider confident
    OOD_ENTROPY_THRESHOLD: float = 2.45          # Entropy threshold for out-of-distribution unsupported classes

    # Model Calibration
    ENABLE_TEMPERATURE_CALIBRATION: bool = True
    DEFAULT_CALIBRATION_TEMPERATURE: float = 1.15

    # Image Quality & Botanical Validation Thresholds
    QUALITY_MIN_RESOLUTION: int = 150
    QUALITY_BLUR_THRESHOLD: float = 85.0         # Laplacian variance threshold
    QUALITY_MIN_BRIGHTNESS: float = 38.0         # Mean grayscale intensity
    QUALITY_MAX_BRIGHTNESS: float = 230.0
    QUALITY_MIN_CONTRAST: float = 28.0           # Grayscale standard deviation
    QUALITY_MIN_VEGETATION_RATIO: float = 0.12   # Minimum leaf area ratio in image
    QUALITY_MAX_BACKGROUND_RATIO: float = 0.85   # Max background before triggering 'leaf too small'
    MULTI_LEAF_CONTOUR_THRESHOLD: int = 3        # Threshold for warning about multiple leaves
    
    # Gemini Vision Hierarchy ("never", "ambiguity_only", "always")
    GEMINI_VISION_MODE: str = os.getenv("GEMINI_VISION_MODE", "always")

    # Resource Protection & Rate Limiting (Free-Tier Safe)
    RATE_LIMIT_PER_MINUTE: int = 60
    MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
    MAX_CONCURRENT_INFERENCES: int = 4
    REQUEST_TIMEOUT_SECONDS: float = 30.0

    class Config:
        env_file = str(PROJECT_ROOT / ".env")
        extra = "ignore"

settings = Settings()
