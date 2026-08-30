"""
Gemini AI Explanation & Multimodal Verification Layer for PlantCare
Provides natural-language interpretations, care advice, explanation caching,
and conditional multimodal vision verification according to a strict hierarchy.
"""

import json
from io import BytesIO
from typing import Optional, Dict, Any
from PIL import Image

from app.core.config import settings
from app.schemas.analysis import GeminiExplanation
from app.schemas.disease import DiseaseInfo

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
                self._client = genai.GenerativeModel("gemini-1.5-flash")
                print("Gemini AI service initialized successfully.")
            except Exception as e:
                print(f"Failed to initialize Gemini AI: {e}")
                self._client = None
        else:
            print("No GEMINI_API_KEY provided; operating in local fallback mode.")

    def should_verify_with_vision(self, plant_val_status: str, plant_confidence: float) -> bool:
        """
        Determines whether Gemini Vision should be invoked based on the configured validation hierarchy:
        1. "never": Never call Gemini Vision (purely local verification)
        2. "ambiguity_only": Call only when status is 'warning' or confidence is borderline
        3. "always": Call on all images if client available
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
            response = self._client.generate_content([prompt, pil_image])
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
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
        # 1. Check in-memory cache
        cache_key = f"{plant}_{predicted_condition}_{int(confidence_percent // 10)}_{state}"
        if cache_key in self._explanation_cache:
            return self._explanation_cache[cache_key]

        # 2. Case: Uncertain condition or unsupported disease state
        if state == "plant_uncertain":
            summary = f"A botanical {plant} specimen was recognized, but the model has uncertainty between multiple candidate conditions."
            interpretation = "Diffuse lesion patterns, uneven lighting, or overlapping leaves can create ambiguity across similar fungal/bacterial symptoms."
            care_rec = "1. Re-photograph a single flat leaf with direct natural lighting.\n2. Inspect both top and underside of leaves for distinct fungal spores or concentric rings."
            exp = GeminiExplanation(
                summary=summary,
                interpretation=interpretation,
                care_recommendation=care_rec,
                powered_by_gemini=False
            )
            self._explanation_cache[cache_key] = exp
            return exp

        if state == "plant_unsupported_condition":
            summary = f"Plant specimen recognized ({plant}), but this specific disease appears outside PlantCare's 21 supported conditions."
            interpretation = "The leaf exhibits visual stress or chlorosis that does not confidently map to any known benchmark disease pattern."
            care_rec = "1. Consult local agricultural extension services for specialized laboratory tissue culture testing.\n2. Ensure soil moisture and balanced N-P-K nutrient feeding."
            exp = GeminiExplanation(
                summary=summary,
                interpretation=interpretation,
                care_recommendation=care_rec,
                powered_by_gemini=False
            )
            self._explanation_cache[cache_key] = exp
            return exp

        # 3. Try Gemini API for known conditions if client available
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
  "care_recommendation": "2 concise bulleted actionable next steps for the grower."
}}
Output raw valid JSON only without markdown formatting.
"""
                response = self._client.generate_content(prompt)
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
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
                print(f"Gemini API generation failed ({e}); using local synthesis fallback.")

        # 4. Curated local fallback synthesis
        if disease_info and disease_info.is_healthy:
            summary = f"The {plant} specimen appears in excellent health with no noticeable pathogen lesions."
            interpretation = "Vibrant green coloration and intact leaf margin structure indicate balanced irrigation and favorable growing conditions."
            care_rec = "Continue regular ground-level watering and weekly monitoring to sustain peak vegetative vigor."
        elif disease_info:
            summary = f"{predicted_condition} identified on {plant} with {confidence_percent:.1f}% confidence. Severity level is {disease_info.severity.lower()}."
            interpretation = f"Infection is typically driven by {', '.join(disease_info.causes[:2]) if disease_info.causes else 'environmental moisture'}. Look for {disease_info.symptoms[0] if disease_info.symptoms else 'leaf spotting'}."
            imm_step = disease_info.treatment.immediate_steps[0] if disease_info.treatment.immediate_steps else "Prune affected leaves."
            org_step = disease_info.treatment.organic_options[0] if disease_info.treatment.organic_options else "Apply preventive bio-fungicide."
            care_rec = f"1. {imm_step}\n2. {org_step}"
        else:
            summary = f"Condition diagnosed as {predicted_condition} on {plant} ({confidence_percent:.1f}% confidence)."
            interpretation = "Symptoms are consistent with localized foliar tissue discoloration."
            care_rec = "1. Isolate the affected plant.\n2. Monitor for further spreading to adjacent foliage."

        exp = GeminiExplanation(
            summary=summary,
            interpretation=interpretation,
            care_recommendation=care_rec,
            powered_by_gemini=False
        )
        self._explanation_cache[cache_key] = exp
        return exp

gemini_service = GeminiService()
