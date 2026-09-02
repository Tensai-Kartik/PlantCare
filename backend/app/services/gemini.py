"""
Gemini AI Explanation & Multimodal Verification Layer for PlantCare
Engineered for high free-tier efficiency, quota protection, multi-model fallback chain,
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
        self._models_pool: Dict[str, Any] = {}
        self._primary_model_name: str = settings.GEMINI_MODEL
        self._fallback_model_names: List[str] = [
            m.strip() for m in settings.GEMINI_FALLBACK_MODELS.split(",") if m.strip()
        ]
        if self._primary_model_name not in self._fallback_model_names:
            self._fallback_model_names.insert(0, self._primary_model_name)

        self._model_cooldowns: Dict[str, float] = {}
        self._explanation_cache: Dict[str, GeminiExplanation] = {}
        self._vision_cache: Dict[str, Dict[str, Any]] = {}
        
        # Sliding window rate limiter (free tier safety: max 15 requests / 60 seconds)
        self._max_rpm: int = 15
        self._call_timestamps: deque = deque()
        
        self._init_client()

    def _init_client(self):
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                
                generation_config = {
                    "temperature": 0.2,
                    "top_p": 0.8,
                    "top_k": 20,
                    "max_output_tokens": 800,
                }
                
                for m_name in self._fallback_model_names:
                    try:
                        self._models_pool[m_name] = genai.GenerativeModel(
                            model_name=m_name,
                            generation_config=generation_config
                        )
                    except Exception as me:
                        print(f"Failed to register model {m_name}: {me}")
                        
                print(f"Gemini AI service initialized with model pool: {list(self._models_pool.keys())}")
            except Exception as e:
                print(f"Failed to initialize Gemini AI: {e}")
                self._models_pool = {}
        else:
            print("No GEMINI_API_KEY provided; operating in local fallback mode.")

    def is_available(self) -> bool:
        if not self._models_pool or not self.api_key:
            return False
        # Check if at least one model is not in cooldown
        now = time.time()
        for m_name in self._models_pool:
            if now >= self._model_cooldowns.get(m_name, 0.0):
                return True
        return False

    def _get_active_model(self) -> Tuple[Optional[str], Optional[Any]]:
        """Returns the first available model that is not currently in rate limit cooldown."""
        now = time.time()
        for m_name in self._fallback_model_names:
            if m_name in self._models_pool:
                if now >= self._model_cooldowns.get(m_name, 0.0):
                    return m_name, self._models_pool[m_name]
        return None, None

    def _record_model_error(self, model_name: str, err: Exception):
        """Sets temporary cooldown on specific model if 429 quota or 504 timeout occurs."""
        err_str = str(err).lower()
        if "429" in err_str or "quota" in err_str or "resourceexhausted" in err_str:
            self._model_cooldowns[model_name] = time.time() + 20.0
            print(f"Gemini Quota Notice ({model_name}): 429 received. Cooldown set for 20s. Switching to fallback.")
        elif "504" in err_str or "deadline" in err_str:
            self._model_cooldowns[model_name] = time.time() + 10.0
            print(f"Gemini Timeout Notice ({model_name}): 504 Deadline. Cooldown set for 10s. Switching to fallback.")

    def _check_and_record_rate_limit(self) -> bool:
        now = time.time()
        while self._call_timestamps and self._call_timestamps[0] <= now - 60.0:
            self._call_timestamps.popleft()
            
        if len(self._call_timestamps) >= self._max_rpm:
            print(f"Gemini Rate Limiter: {len(self._call_timestamps)} calls in last 60s. Throttling.")
            return False
            
        self._call_timestamps.append(now)
        return True

    @staticmethod
    def _optimize_image_for_api(image_bytes: bytes, max_dim: int = 640, quality: int = 85) -> Tuple[Image.Image, str]:
        """
        Compresses and resizes image before sending to Gemini API.
        Preserves foliar lesion details while keeping payload compact.
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
        with SHA-256 caching and multi-model fallback chain.
        """
        if not self.is_available():
            return None

        # 1. Optimize image and check cache
        try:
            pil_image, img_hash = self._optimize_image_for_api(image_bytes, max_dim=640, quality=85)
            if img_hash in self._vision_cache:
                print(f"Gemini Cache Hit (SHA-256: {img_hash[:8]}): Serving cached vision result.")
                return self._vision_cache[img_hash]
        except Exception as e:
            print(f"Image preprocessing warning: {e}")
            pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
            img_hash = None

        if not self._check_and_record_rate_limit():
            return None

        prompt = """You are an expert plant pathologist, botanist, and agronomist for PlantCare AI.
Analyze this plant leaf image carefully and identify:
1. The exact plant host species and common botanical name (e.g., Tomato, Potato, Apple, Grape, Pepper, Corn, Rose, Mango, Lemon, Rice, Wheat, Cotton, Strawberry, Monstera, Basil, Citrus, Hibiscus, etc.).
2. The specific health condition, fungal/bacterial/viral disease, pest damage, nutrient deficiency, or confirm if the leaf is Healthy.
3. Botanical scientific name and pathogen causal agent if applicable.
4. Visual indicators observed (lesion spots, concentric rings, chlorosis, necrosis, mosaic patterns, pustules, wilting, etc.).
5. Structured treatment protocol (immediate action, organic options, conventional chemical options) and prevention practices.

Respond ONLY in valid raw JSON with this EXACT structure (do not include extra text outside JSON):
{
  "plant": "Common Plant Name (e.g. Tomato)",
  "scientific_name": "Botanical Latin species name (e.g. Solanum lycopersicum)",
  "condition_name": "Disease or Condition Name (e.g. Early Blight, Black Spot, Nitrogen Deficiency, Healthy)",
  "is_healthy": false,
  "severity": "Low / Moderate / High / Critical / Healthy",
  "confidence_percent": 95.0,
  "symptoms": [
    "Concentric dark brown rings on lower foliage",
    "Yellow chlorotic halo surrounding necrotic lesions",
    "Premature leaf drop"
  ],
  "causes": [
    "Alternaria solani fungal spores",
    "Warm humid weather and prolonged leaf wetness"
  ],
  "treatment": {
    "immediate_steps": [
      "Prune and safely discard all infected lower foliage",
      "Avoid overhead watering; irrigate strictly at the soil base"
    ],
    "organic_options": [
      "Apply copper octanoate or Bacillus subtilis bio-fungicide",
      "Spray cold-pressed neem oil at 7-day intervals"
    ],
    "conventional_options": [
      "Apply chlorothalonil or mancozeb protectant fungicide",
      "Rotate with azoxystrobin to prevent pathogen resistance"
    ]
  },
  "prevention": [
    "Maintain 24-36 inch crop spacing for optimal airflow",
    "Apply organic straw mulch to prevent soil-splash spore dispersal",
    "Practice 3-year crop rotation with non-solanaceous plants"
  ],
  "important_notes": [
    "Fungal spores can overwinter in plant debris; sanitize all gardening shears."
  ],
  "agronomist_summary": "Pathology analysis confirms Early Blight on Tomato foliage with distinctive target-spot lesions. Immediate pruning and foliar fungicide application recommended to protect yield."
}"""

        # Try models in fallback pool
        for m_name in self._fallback_model_names:
            if m_name not in self._models_pool:
                continue
            if time.time() < self._model_cooldowns.get(m_name, 0.0):
                continue

            model_client = self._models_pool[m_name]
            try:
                print(f"Calling Gemini Vision using model '{m_name}'...")
                response = model_client.generate_content([prompt, pil_image])
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.startswith("```"):
                    text = text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                data = json.loads(text.strip())
                
                # Normalize and ensure key fields exist
                if not data.get("plant"):
                    data["plant"] = "Plant"
                if not data.get("condition_name"):
                    data["condition_name"] = "Healthy" if data.get("is_healthy", False) else "Undetermined Condition"
                if "confidence_percent" not in data or not isinstance(data["confidence_percent"], (int, float)):
                    data["confidence_percent"] = 95.0
                
                data["model_used"] = m_name
                print(f"Gemini Vision Success via {m_name}: {data.get('plant')} - {data.get('condition_name')} ({data.get('confidence_percent')}%)")

                if img_hash:
                    self._vision_cache[img_hash] = data
                    
                return data
            except Exception as e:
                self._record_model_error(m_name, e)
                print(f"Gemini Vision call with model '{m_name}' encountered: {e}")

        print("All Gemini Vision models in pool failed or are in cooldown; falling back to local ensemble.")
        return None

    def verify_image_is_plant(self, image_bytes: bytes) -> Optional[Dict[str, Any]]:
        """
        Uses Gemini Vision multimodal API to verify ambiguous images.
        """
        if not self.is_available():
            return None

        if not self._check_and_record_rate_limit():
            return None

        try:
            pil_image, _ = self._optimize_image_for_api(image_bytes, max_dim=384, quality=75)

            prompt = """You are an expert Computer Vision agronomist for PlantCare.
Analyze this image. Determine if the primary subject is a real botanical plant, crop, leaf, flower, tree, or agricultural produce.
If it is a vehicle, animal, electronic device, person, indoor furniture, or other non-plant object, mark is_plant as false.

Respond in raw JSON only with EXACTLY this structure:
{
  "is_plant": true,
  "subject_category": "plant",
  "identified_subject": "e.g. Tomato leaf, Sports car, Laptop computer",
  "confidence": 0.95,
  "reason": "1 concise sentence explaining the subject."
}
"""
            for m_name in self._fallback_model_names:
                if m_name not in self._models_pool or time.time() < self._model_cooldowns.get(m_name, 0.0):
                    continue
                try:
                    client = self._models_pool[m_name]
                    response = client.generate_content([prompt, pil_image])
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
                    self._record_model_error(m_name, e)
                    print(f"Gemini plant check failed on {m_name}: {e}")
        except Exception as e:
            print(f"Gemini plant check exception: {e}")

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
            prompt = f"""You are an expert agricultural plant pathologist for PlantCare.
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
  "interpretation": "2 sentences explaining why these symptoms occur on {plant}.",
  "care_recommendation": "Bullet points with actionable organic and conventional steps."
}}
"""
            for m_name in self._fallback_model_names:
                if m_name not in self._models_pool or time.time() < self._model_cooldowns.get(m_name, 0.0):
                    continue
                try:
                    client = self._models_pool[m_name]
                    response = client.generate_content(prompt)
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
                    self._record_model_error(m_name, e)
                    print(f"Gemini API explanation generation failed on {m_name}: {e}")

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
