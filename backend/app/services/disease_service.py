"""
Disease Knowledge Base Service for PlantCare
Manages pathology data, symptom indices, treatment steps, search, and filtering.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict

from app.core.config import settings
from app.schemas.disease import DiseaseInfo, TreatmentDetails, DiseaseListResponse

class DiseaseService:
    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or settings.DISEASE_DATA_PATH
        self._diseases_by_id: Dict[str, DiseaseInfo] = {}
        self._all_diseases: List[DiseaseInfo] = []
        self._plants: List[str] = []
        self.load_data()

    def load_data(self):
        if not self.data_path.exists():
            print(f"Warning: Disease data file not found at {self.data_path}")
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self._all_diseases = []
        self._diseases_by_id = {}
        plants_set = set()

        for item in raw_data.get("diseases", []):
            treatment_dict = item.get("treatment", {})
            treatment_obj = TreatmentDetails(
                immediate_steps=treatment_dict.get("immediate_steps", []),
                organic_options=treatment_dict.get("organic_options", []),
                conventional_options=treatment_dict.get("conventional_options", [])
            )
            d_info = DiseaseInfo(
                id=item["id"],
                name=item["name"],
                scientific_name=item.get("scientific_name"),
                plant=item.get("plant", "Unknown"),
                is_healthy=item.get("is_healthy", False),
                description=item.get("description", ""),
                symptoms=item.get("symptoms", []),
                causes=item.get("causes", []),
                severity=item.get("severity", "Moderate"),
                spread=item.get("spread"),
                image_url=item.get("image_url", f"/examples/{item['id']}.jpg"),
                treatment=treatment_obj,
                prevention=item.get("prevention", []),
                important_notes=item.get("important_notes", [])
            )
            self._diseases_by_id[d_info.id] = d_info
            self._all_diseases.append(d_info)
            plants_set.add(d_info.plant)

        self._plants = sorted(list(plants_set))
        print(f"Loaded {len(self._all_diseases)} diseases across {len(self._plants)} plant categories.")

    def get_by_id(self, disease_id: str) -> Optional[DiseaseInfo]:
        if disease_id in self._diseases_by_id:
            return self._diseases_by_id[disease_id]

        # Normalized search fallback
        norm_id = disease_id.lower().replace(" ", "_").replace("-", "_")
        if norm_id in self._diseases_by_id:
            return self._diseases_by_id[norm_id]

        # Generate sensible default if not found
        return self._generate_fallback(disease_id)

    def list_all(self, plant: Optional[str] = None, severity: Optional[str] = None, query: Optional[str] = None) -> DiseaseListResponse:
        results = self._all_diseases

        if plant and plant.lower() != "all":
            results = [d for d in results if d.plant.lower() == plant.lower()]

        if severity and severity.lower() != "all":
            results = [d for d in results if d.severity.lower() == severity.lower()]

        if query:
            q = query.lower()
            results = [
                d for d in results
                if q in d.name.lower() or
                   (d.scientific_name and q in d.scientific_name.lower()) or
                   q in d.plant.lower() or
                   q in d.description.lower() or
                   any(q in s.lower() for s in d.symptoms)
            ]

        return DiseaseListResponse(
            total=len(results),
            plants=self._plants,
            diseases=results
        )

    def get_plants(self) -> List[str]:
        return self._plants

    def _generate_fallback(self, class_id: str) -> DiseaseInfo:
        human_name = class_id.replace("_", " ").title()
        is_healthy = "healthy" in class_id.lower()
        plant = class_id.split("_")[0].capitalize()

        return DiseaseInfo(
            id=class_id,
            name=human_name,
            scientific_name="Pathogen unidentified",
            plant=plant,
            is_healthy=is_healthy,
            description=f"Pathology information for {human_name}. Regular monitoring is advised.",
            symptoms=["Visible discoloration or textural variations on leaf blade."],
            causes=["Biotic or abiotic agricultural stress factors."],
            severity="Healthy" if is_healthy else "Moderate",
            spread="Unknown",
            image_url=f"/examples/{class_id}.jpg",
            treatment=TreatmentDetails(
                immediate_steps=["Isolate affected plant part to prevent transmission."],
                organic_options=["Apply protective neem oil or biofungicide spray."],
                conventional_options=["Consult local agricultural extension for registered chemicals."]
            ),
            prevention=["Maintain clean cultural practices and dry foliage."],
            important_notes=["Ensure proper diagnosis before intensive pesticide application."]
        )

disease_service = DiseaseService()
