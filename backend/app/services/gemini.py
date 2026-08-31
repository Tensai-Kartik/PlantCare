"""
Gemini AI Explanation & Multimodal Verification Layer for PlantCare
Provides natural-language interpretations, care advice, explanation caching,
and simultaneous multimodal vision verification and cross-referencing.
"""

import json
from io import BytesIO
from typing import Optional, Dict, Any, List
from PIL import Image

from app.core.config import settings
from app.schemas.analysis import GeminiExplanation
from app.schemas.disease import DiseaseInfo, TreatmentGuide

class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self._client = None
        self._explanation_cache: Dict[str, GeminiExplanation] = {}
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                self._client = genai.GenerativeModel(settings.GEMINI_MODEL)
                print(f"Gemini AI service ({settings.GEMINI_MODEL}) initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Gemini AI: {e}")
                self._client = None
        else:
            print("No GEMINI_API_KEY provided; operating in local fallback mode.")

    def is_available(self) -> bool:
        return self._client is not None

    def should_verify_with_vision(self, plant_val_status: str, plant_confidence: float) -> bool:
        """
        Determines whether Gemini Vision should be invoked based on the configured validation hierarchy.
        """
        if not self._client:
            return False

        mode = settings.GEMINI_VISION_MODE.lower()
        if mode == "never":
            return False
        elif mode == "always":
            return True
        else:  # "ambiguity_only"
            return plant_val_status == "warning" or (40.0 <= plant_confidence <= 75.0)

    def analyze_plant_multimodal(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Executes zero-shot multimodal agricultural pathology diagnosis on the plant leaf image.
        """
        if not self._client:
            return None

        try:
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            prompt = """You are an expert plant pathologist and AI agronomist for PlantCare.
Analyze this leaf image carefully. Identify the host plant species and diagnose the exact health condition, disease, pest, or nutrient deficiency.
Respond ONLY in valid raw JSON with this exact schema:
{
  "plant": "Host plant common name (e.g. Tomato, Potato, Apple, Rose, Mango, Lemon, Corn, Grape, Monstera)",
  "scientific_name": "Botanical species name if known",
  "condition_name": "Diagnosed condition (e.g. Early Blight, Late Blight, Black Spot, Anthracnose, Nitrogen Deficiency, Healthy)",
  "is_healthy": false,
  "severity": "Low / Moderate / High / Critical / Healthy",
  "confidence_percent": 95.0,
  "symptoms": ["Symptom 1 with visual lesion description", "Symptom 2", "Symptom 3"],
  "causes": ["Pathogen or environmental trigger 1", "Trigger 2"],
  "treatment": {
    "immediate_steps": ["Action step 1", "Action step 2"],
    "organic_options": ["Organic low-impact remedy 1", "Organic remedy 2"],
    "conventional_options": ["Standard agricultural treatment 1", "Treatment 2"]
  },
  "prevention": ["Cultural prevention practice 1", "Prevention practice 2"],
  "important_notes": ["Agronomic advisory note 1"],
  "agronomist_summary": "1-2 concise sentences summarizing the diagnosis and urgency."
}"""

            response = self._client.generate_content([prompt, pil_image], request_options={"timeout": 6.0})
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text.strip())
            return data
        except Exception as e:
            print(f"Gemini multimodal vision diagnosis error: {e}")
            return None

    def verify_image_is_plant(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Uses Gemini Vision multimodal API to verify ambiguous images.
        """
        if not self._client:
            return None

        try:
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")

            prompt = """You are an expert Computer Vision agronomist for PlantCare.
Analyze this image. Determine if the primary subject is a real plant, crop, leaf, flower, or agricultural produce.
If it is a car, vehicle, animal, electronic device, person, indoor furniture, or other non-plant object, mark is_plant as false.

Respond in raw JSON only with EXACTLY this structure:
{
  "is_plant": true,
  "subject_category": "plant",
  "identified_subject": "e.g. Tomato leaf, Sports car, Laptop computer",
  "confidence": 0.95,
  "reason": "1 concise sentence explaining the subject and whether it is a plant specimen."
}
"""
            response = self._client.generate_content([prompt, pil_image], request_options={"timeout": 6.0})
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text.strip())
            return data
        except Exception as e:
            print(f"Gemini Vision validation call failed: {e}")
            return None

    def generate_explanation(
        self,
        plant: str,
        predicted_condition: str,
        confidence_percent: float,
        disease_info: Optional[DiseaseInfo] = None,
        state: str = "known_high"
    ) -> GeminiExplanation:
        """
        Generates a concise, structured AI pathology explanation with in-memory caching.
        Uses Gemini if configured, otherwise synthesizes from the local curated pathology knowledge base.
        """
        cache_key = f"{plant}_{predicted_condition}_{int(confidence_percent // 10)}_{state}"
        if cache_key in self._explanation_cache:
            return self._explanation_cache[cache_key]

        if self._client and disease_info:
            try:
                prompt = f"""
You are an expert agricultural plant pathologist for PlantCare.
Analyze the following computer vision classification result:
- Host Plant: {plant}
- Detected Condition: {predicted_condition}
- Model Confidence: {confidence_percent:.1f}%
- Disease Severity: {disease_info.severity}
- Primary Symptoms: {', '.join(disease_info.symptoms[:3])}
- Known Causes: {', '.join(disease_info.causes[:2])}
- Recommended Immediate Actions: {', '.join(disease_info.treatment.immediate_steps[:2])}

Generate a concise, professional JSON response with EXACTLY the following format:
{{
  "summary": "1-2 sentences summarizing the diagnosis and urgency.",
  "interpretation": "2 sentences explaining why these symptoms occur and the environmental triggers.",
  "care_recommendation": "Bullet points with actionable organic and conventional steps."
}}
"""
                response = self._client.generate_content(prompt, request_options={"timeout": 6.0})
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                data = json.loads(text.strip())
                exp = GeminiExplanation(
                    summary=data.get("summary", ""),
                    interpretation=data.get("interpretation", ""),
                    care_recommendation=data.get("care_recommendation", ""),
                    powered_by_gemini=True
                )
                self._explanation_cache[cache_key] = exp
                return exp
            except Exception as e:
                print(f"Gemini API explanation generation failed: {e}")

        # Local synthesis fallback
        symptoms_text = f"Visual indicators include {', '.join(disease_info.symptoms[:2])}." if disease_info else "Foliar discoloration detected."
        causes_text = f"Commonly triggered by {', '.join(disease_info.causes[:2])}." if disease_info else "Environmental stress or pathogen infection."
        care_text = "\n".join([f"• {step}" for step in (disease_info.treatment.immediate_steps[:2] if disease_info else ["Isolate plant", "Prune damaged foliage"])])

        exp = GeminiExplanation(
            summary=f"Pathology scan indicates {predicted_condition} on {plant} foliage with {confidence_percent:.1f}% confidence.",
            interpretation=f"{symptoms_text} {causes_text}",
            care_recommendation=care_text,
            powered_by_gemini=False
        )
        self._explanation_cache[cache_key] = exp
        return exp

gemini_service = GeminiService()
