"""
Gemini AI Explanation & Multimodal Verification Layer for PlantCare
Engineered for high free-tier efficiency, quota protection, token minimization,
SHA-256 content caching, adaptive image downsampling, and sliding-window rate throttling.
"""

import hashlib
import json
import time
from collections import deque
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
from PIL import Image

from app.core.config import settings
from app.schemas.analysis import GeminiExplanation
from app.schemas.disease import DiseaseInfo, TreatmentGuide

class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self._client = None
        self._explanation_cache: Dict[str, GeminiExplanation] = {}
        self._vision_cache: Dict[str, Dict[str, Any]] = {}
        
        # Sliding window rate limiter (free tier safety: max 12 requests / 60 seconds)
        self._max_rpm: int = 12
        self._call_timestamps: deque = deque()
        self._cooldown_until: float = 0.0  # Temporary cooldown if 429 encountered
        
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                
                # Optimized generation config for low token consumption & fast latency
                generation_config = {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_output_tokens": 400,
                }
                
                self._client = genai.GenerativeModel(
                    model_name=settings.GEMINI_MODEL,
                    generation_config=generation_config
                )
                print(f"Gemini AI service ({settings.GEMINI_MODEL}) initialized with Quota Optimizer.")
            except Exception as e:
                print(f"Failed to initialize Gemini AI: {e}")
                self._client = None
        else:
            print("No GEMINI_API_KEY provided; operating in local fallback mode.")

    def is_available(self) -> bool:
        if self._client is None:
            return False
        # Check if in temporary cooldown from rate limit
        if time.time() < self._cooldown_until:
            return False
        return True

    def _check_and_record_rate_limit(self) -> bool:
        """
        Sliding-window token bucket to prevent hitting Google AI Studio RPM limits (15 RPM).
        Returns True if safe to proceed, False if throttled.
        """
        now = time.time()
        
        if now < self._cooldown_until:
            return False
            
        # Clean timestamps older than 60 seconds
        while self._call_timestamps and self._call_timestamps[0] <= now - 60.0:
            self._call_timestamps.popleft()
            
        if len(self._call_timestamps) >= self._max_rpm:
            print(f"Gemini Quota Protection: {len(self._call_timestamps)} calls in last 60s. Throttling request to preserve free quota.")
            return False
            
        self._call_timestamps.append(now)
        return True

    def _handle_quota_error(self, err: Exception):
        """
        Sets a 30s automatic cooldown when a 429 quota exhaustion is detected.
        """
        err_str = str(err).lower()
        if "429" in err_str or "quota" in err_str or "exhausted" in err_str:
            self._cooldown_until = time.time() + 30.0
            print("Gemini Quota Protection Activated: Received 429 from Google API. Cooling down for 30s; switching seamlessly to local ensemble.")

    @staticmethod
    def _optimize_image_for_api(image_bytes: bytes, max_dim: int = 512, quality: int = 80) -> Tuple[Image.Image, str]:
        """
        Compresses and resizes image before sending to Gemini API.
        Reduces token payload from ~5MB to ~40KB while preserving fine foliar lesion details.
        Returns the optimized PIL image and its SHA-256 content hash for caching.
        """
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        w, h = pil_image.size
        
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            pil_image = pil_image.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
        optimized_bytes = buffer.getvalue()
        
        img_hash = hashlib.sha256(optimized_bytes).hexdigest()
        buffer.seek(0)
        return Image.open(buffer), img_hash

    def should_verify_with_vision(self, plant_val_status: str, plant_confidence: float) -> bool:
        """
        Determines whether Gemini Vision should be invoked based on the configured validation hierarchy.
        """
        if not self.is_available():
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
        Executes zero-shot multimodal agricultural pathology diagnosis on the plant leaf image
        with SHA-256 caching and downsampling to maximize free-tier rate limits.
        """
        if not self.is_available():
            return None

        # 1. Optimize image and check SHA-256 cache
        try:
            pil_image, img_hash = self._optimize_image_for_api(image_bytes, max_dim=512, quality=80)
            if img_hash in self._vision_cache:
                print(f"Gemini Cache Hit (SHA-256: {img_hash[:8]}): 0 tokens consumed.")
                return self._vision_cache[img_hash]
        except Exception as e:
            print(f"Image preprocessing warning: {e}")
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            img_hash = None

        # 2. Check free-tier rate limit bucket
        if not self._check_and_record_rate_limit():
            return None

        # 3. Concise, token-efficient prompt
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

        try:
            response = self._client.generate_content([prompt, pil_image], request_options={"timeout": 6.0})
            text = response.text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            data = json.loads(text.strip())
            
            # Store in vision cache
            if img_hash:
                self._vision_cache[img_hash] = data
                
            return data
        except Exception as e:
            self._handle_quota_error(e)
            print(f"Gemini multimodal vision diagnosis error: {e}")
            return None

    def verify_image_is_plant(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Uses Gemini Vision multimodal API to verify ambiguous images with rate limiting.
        """
        if not self.is_available():
            return None

        if not self._check_and_record_rate_limit():
            return None

        try:
            pil_image, _ = self._optimize_image_for_api(image_bytes, max_dim=384, quality=75)

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
            response = self._client.generate_content([prompt, pil_image], request_options={"timeout": 5.0})
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
            self._handle_quota_error(e)
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

        if self.is_available() and disease_info and self._check_and_record_rate_limit():
            try:
                prompt = f"""You are an agricultural plant pathologist for PlantCare.
Analyze this diagnosis:
- Host Plant: {plant}
- Detected Condition: {predicted_condition}
- Model Confidence: {confidence_percent:.1f}%
- Disease Severity: {disease_info.severity}
- Primary Symptoms: {', '.join(disease_info.symptoms[:3])}
- Known Causes: {', '.join(disease_info.causes[:2])}
- Recommended Immediate Actions: {', '.join(disease_info.treatment.immediate_steps[:2])}

Generate a concise JSON response with EXACTLY:
{{
  "summary": "1-2 sentences summarizing the diagnosis and urgency.",
  "interpretation": "2 sentences explaining why these symptoms occur.",
  "care_recommendation": "Bullet points with actionable organic and conventional steps."
}}
"""
                response = self._client.generate_content(prompt, request_options={"timeout": 5.0})
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
                self._handle_quota_error(e)
                print(f"Gemini API explanation generation failed: {e}")

        # Local curated synthesis fallback (Zero Token Cost)
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
