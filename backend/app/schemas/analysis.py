"""
Pydantic Schemas for Quality Check, Computer Vision Inference, Grad-CAM, Calibration,
Uncertainty Handling, Model Disagreement, and Performance Auditing.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from .disease import DiseaseInfo

class QualityMetrics(BaseModel):
    width: int
    height: int
    blur_score: float
    brightness: float
    contrast: float
    vegetation_ratio: float
    estimated_leaf_count: int = 1
    background_ratio: float = 0.0

class QualityCheckResult(BaseModel):
    is_suitable: bool
    suitability_score: float = Field(..., ge=0.0, le=100.0)
    status: str  # "suitable", "warning", "rejected"
    is_plant: bool = True
    detected_subject: str = "Plant Leaf / Crop Specimen"
    subject_category: str = "plant"  # "plant", "vehicle", "animal", "electronics", "person", "furniture", "food", "manmade", "non_plant"
    plant_confidence: float = 100.0
    reason_code: Optional[str] = None  # e.g. "SUITABLE_PLANT", "NON_PLANT_OBJECT", "TOO_DARK", "BLURRY", "LEAF_TOO_SMALL", "MULTIPLE_LEAVES"
    warnings: List[str] = Field(default_factory=list)  # e.g. ["LOW_LIGHT", "PARTIAL_LEAF", "MULTIPLE_LEAVES"]
    has_multiple_leaves: bool = False
    leaf_focus_status: str = "optimal"  # "optimal", "leaf_too_small", "excessive_background", "partial_leaf", "obstructed"
    issues: List[str] = Field(default_factory=list)
    positive_indicators: List[str] = Field(default_factory=list)
    metrics: QualityMetrics
    guidance: str

class CandidatePrediction(BaseModel):
    class_id: str
    name: str
    plant: str
    probability: float
    calibrated_probability: float
    probability_percent: float

class PredictionResult(BaseModel):
    class_id: str
    name: str
    scientific_name: Optional[str] = None
    plant: str
    state: str = "known_high"  # "known_high", "known_moderate", "plant_uncertain", "plant_unsupported_condition", "non_plant"
    confidence: float
    raw_confidence: float
    calibrated_confidence: float
    confidence_percent: float
    confidence_level: str  # "High Confidence", "Moderate Confidence", "Low Confidence", "Uncertain Condition", "Outside Supported Scope"
    entropy: float = 0.0
    top1_top2_margin: float = 1.0
    is_healthy: bool = False
    status_message: str = "Known condition identified with high confidence."
    top_candidates: List[CandidatePrediction] = Field(default_factory=list)

class ModelMetadata(BaseModel):
    id: str
    name: str
    architecture: str
    version: str = "1.0.0"
    dataset: str = "PlantVillage + FieldAug"
    dataset_version: str = "2.0"
    image_size: int = 224
    class_count: int = 21
    training_date: str = "2026-08-26"
    temperature: float = 1.15
    ece: Optional[float] = 0.038
    accuracy: Optional[float] = None
    weighted_f1: Optional[float] = None
    latency_ms: Optional[float] = None
    is_default: bool = False

class ModelListResponse(BaseModel):
    models: List[ModelMetadata]
    default: str

class GeminiExplanation(BaseModel):
    summary: str
    interpretation: str
    care_recommendation: str
    powered_by_gemini: bool = False

class NonPlantDetails(BaseModel):
    detected_subject: str
    category: str
    confidence_percent: float
    message: str
    suggestions: List[str] = Field(default_factory=list)

class ModelComparisonEntry(BaseModel):
    model_id: str
    model_name: str
    predicted_class_id: str
    predicted_name: str
    confidence_percent: float

class ModelDisagreementResult(BaseModel):
    enabled: bool = False
    agreement_status: str = "AGREED"  # "AGREED", "DISAGREED", "PARTIAL"
    models_agree: bool = True
    consensus_prediction: Optional[str] = None
    message: str = "Models agree on condition diagnosis."
    comparison: List[ModelComparisonEntry] = Field(default_factory=list)

class PerformanceMetrics(BaseModel):
    image_validation_ms: float = 0.0
    preprocessing_ms: float = 0.0
    model_inference_ms: float = 0.0
    gradcam_ms: float = 0.0
    disease_metadata_lookup_ms: float = 0.0
    gemini_ms: float = 0.0
    total_request_ms: float = 0.0

class PredictionAudit(BaseModel):
    request_id: str
    model_id: str
    model_version: str
    prediction_state: str
    raw_confidence: float
    calibrated_confidence: float
    temperature_applied: float
    entropy: float
    top1_top2_margin: float
    suitability_score: float
    validator_status: str
    reason_code: Optional[str] = None
    gradcam_status: str = "not_generated"
    gemini_status: str = "disabled"
    performance_metrics: PerformanceMetrics

class AnalysisResponse(BaseModel):
    success: bool
    is_plant: bool = True
    prediction: Optional[PredictionResult] = None
    model: Optional[ModelMetadata] = None
    quality: QualityCheckResult
    disease: Optional[DiseaseInfo] = None
    non_plant_details: Optional[NonPlantDetails] = None
    explanation: Optional[GeminiExplanation] = None
    gradcam_heatmap_base64: Optional[str] = None
    comparison: Optional[ModelDisagreementResult] = None
    audit: Optional[PredictionAudit] = None
    timestamp: str

class ExampleLeaf(BaseModel):
    id: str
    title: str
    plant: str
    condition: str
    image_url: str
    is_healthy: bool = False
