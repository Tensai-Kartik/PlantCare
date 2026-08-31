"""
Disease Pathology and Treatment Pydantic Schemas
"""

from typing import List, Optional
from pydantic import BaseModel, Field

class TreatmentDetails(BaseModel):
    immediate_steps: List[str] = Field(default_factory=list, description="Emergency actions to stop spread")
    organic_options: List[str] = Field(default_factory=list, description="Eco-friendly / bio-control remedies")
    conventional_options: List[str] = Field(default_factory=list, description="Standard agricultural chemicals / fungicides")

# Alias for compatibility
TreatmentGuide = TreatmentDetails

class DiseaseInfo(BaseModel):
    id: str
    name: str
    scientific_name: Optional[str] = None
    plant: str
    is_healthy: bool = False
    description: str
    symptoms: List[str] = Field(default_factory=list)
    causes: List[str] = Field(default_factory=list)
    severity: str = "Moderate"  # Healthy, Low, Moderate, High, Critical
    spread: Optional[str] = None
    image_url: Optional[str] = None
    treatment: TreatmentDetails
    prevention: List[str] = Field(default_factory=list)
    important_notes: List[str] = Field(default_factory=list)
    favorable_conditions: Optional[str] = None

class DiseaseListResponse(BaseModel):
    total: int
    plants: List[str]
    diseases: List[DiseaseInfo]
