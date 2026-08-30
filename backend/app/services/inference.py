"""
Inference Orchestration Service for PlantCare
Coordinates plant presence validation, micro-timing performance metrics,
temperature confidence calibration, Shannon entropy uncertainty analysis,
Grad-CAM heatmaps, disease knowledge mapping, multi-model disagreement comparison,
and structured prediction auditing.
"""

import time
import uuid
from datetime import datetime
from io import BytesIO
from typing import Optional, List
from PIL import Image
import torch
import torch.nn.functional as F

from app.core.config import settings
from app.schemas.analysis import (
    AnalysisResponse,
    PredictionResult,
    CandidatePrediction,
    QualityCheckResult,
    NonPlantDetails,
    PerformanceMetrics,
    PredictionAudit,
    ModelDisagreementResult
)
from app.services.image_quality import image_quality_service
from app.services.model_registry import model_registry
from app.services.calibration import calibration_service
from app.services.gradcam import gradcam_service
from app.services.disease_service import disease_service
from app.services.gemini import gemini_service

class InferenceService:
    def __init__(self):
        pass

    def run_analysis(
        self,
        image_bytes: bytes,
        model_id: Optional[str] = None,
        skip_quality_check: bool = False,
        enable_model_comparison: bool = False
    ) -> AnalysisResponse:
        """
        Executes end-to-end plant disease diagnosis workflow with calibration,
        uncertainty categorization, Grad-CAM, stage micro-timings, and audit metadata.
        """
        total_start = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:10]}"

        # Stage 1: Image Quality & Multi-Signal Plant Presence Validation
        val_start = time.perf_counter()
        quality_result: QualityCheckResult = image_quality_service.evaluate_image(image_bytes)
        val_end = time.perf_counter()
        validation_ms = (val_end - val_start) * 1000.0

        # Out-of-Domain Guardrail: Halt if image is verified as non-plant
        if not quality_result.is_plant:
            total_end = time.perf_counter()
            total_ms = (total_end - total_start) * 1000.0
            
            perf = PerformanceMetrics(
                image_validation_ms=round(validation_ms, 2),
                preprocessing_ms=0.0,
                model_inference_ms=0.0,
                gradcam_ms=0.0,
                disease_metadata_lookup_ms=0.0,
                gemini_ms=0.0,
                total_request_ms=round(total_ms, 2)
            )

            audit = PredictionAudit(
                request_id=req_id,
                model_id=model_id or model_registry.default_model_id,
                model_version="1.2.0",
                prediction_state="non_plant",
                raw_confidence=0.0,
                calibrated_confidence=0.0,
                temperature_applied=1.0,
                entropy=0.0,
                top1_top2_margin=0.0,
                suitability_score=quality_result.suitability_score,
                validator_status=quality_result.status,
                reason_code=quality_result.reason_code,
                gradcam_status="halted_non_plant",
                gemini_status="skipped_non_plant",
                performance_metrics=perf
            )

            non_plant_details = NonPlantDetails(
                detected_subject=quality_result.detected_subject,
                category=quality_result.subject_category,
                confidence_percent=round(100.0 - quality_result.plant_confidence, 1),
                message=f"PlantCare has detected that this image contains a '{quality_result.detected_subject}' rather than a botanical plant or crop specimen.",
                suggestions=[
                    "Take a clear, close-up photo focused on a single plant leaf.",
                    "Ensure the leaf foliage fills at least 30% to 70% of the camera frame.",
                    "Avoid uploading photos of vehicles, pets, electronics, human portraits, or manmade objects.",
                    "Use well-balanced natural lighting and ensure the leaf veins/lesions are in sharp focus."
                ]
            )

            return AnalysisResponse(
                success=False,
                is_plant=False,
                prediction=None,
                model=None,
                quality=quality_result,
                disease=None,
                non_plant_details=non_plant_details,
                explanation=None,
                gradcam_heatmap_base64=None,
                comparison=None,
                audit=audit,
                timestamp=datetime.utcnow().isoformat() + "Z"
            )

        # Stage 2: Preprocessing
        prep_start = time.perf_counter()
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        model_wrapper = model_registry.get_model(model_id)
        input_tensor = model_wrapper.preprocess(pil_image)
        prep_end = time.perf_counter()
        preprocessing_ms = (prep_end - prep_start) * 1000.0

        # Stage 3: Forward Pass & Temperature Calibration
        inf_start = time.perf_counter()
        model_wrapper.model.eval()
        with torch.no_grad():
            logits = model_wrapper.model(input_tensor)

        # Apply Temperature Scaling for Calibrated Confidence
        temp = model_wrapper.temperature if settings.ENABLE_TEMPERATURE_CALIBRATION else 1.0
        raw_probs, calibrated_probs = calibration_service.calibrate_logits(logits, temperature=temp)
        inf_end = time.perf_counter()
        inference_ms = (inf_end - inf_start) * 1000.0

        # Top Candidates Computation
        classes = model_wrapper.classes
        top_k = min(5, len(classes))
        top_probs, top_indices = torch.topk(calibrated_probs, top_k)
        raw_top_probs, _ = torch.topk(raw_probs, top_k)

        top_class_idx = top_indices[0].item()
        calibrated_top_prob = top_probs[0].item()
        raw_top_prob = raw_top_probs[0].item()
        top_class_id = classes[top_class_idx]

        # Uncertainty Metrics: Shannon Entropy & Top-1 vs Top-2 Margin
        entropy = calibration_service.compute_entropy(calibrated_probs)
        top_margin = calibration_service.compute_top_margin(calibrated_probs)

        # 5-State Decision Classification
        pred_state, conf_level, status_msg = calibration_service.categorize_prediction_state(
            calibrated_prob=calibrated_top_prob,
            raw_prob=raw_top_prob,
            entropy=entropy,
            top_margin=top_margin,
            num_classes=len(classes),
            is_plant=True
        )

        # Build Top Candidates List
        candidates: List[CandidatePrediction] = []
        for i in range(top_k):
            idx = top_indices[i].item()
            c_prob = top_probs[i].item()
            r_prob = raw_probs[idx].item()
            c_id = classes[idx]
            plant_name = c_id.split("_")[0].capitalize()
            candidates.append(CandidatePrediction(
                class_id=c_id,
                name=c_id.replace("_", " ").title(),
                plant=plant_name,
                probability=round(r_prob, 4),
                calibrated_probability=round(c_prob, 4),
                probability_percent=round(c_prob * 100.0, 1)
            ))

        # Stage 4: Disease Knowledge Retrieval
        lookup_start = time.perf_counter()
        disease_info = disease_service.get_by_id(top_class_id)
        is_healthy = disease_info.is_healthy if disease_info else ("healthy" in top_class_id.lower())
        lookup_end = time.perf_counter()
        lookup_ms = (lookup_end - lookup_start) * 1000.0

        # Stage 5: Grad-CAM Heatmap Generation (Only if confident / moderate)
        gradcam_start = time.perf_counter()
        gradcam_b64 = None
        gradcam_status = "skipped_unsupported" if pred_state == "plant_unsupported_condition" else "generated"
        if pred_state != "plant_unsupported_condition":
            try:
                target_layer = model_wrapper.get_target_layer_for_gradcam()
                gradcam_b64 = gradcam_service.generate_gradcam(
                    model=model_wrapper.model,
                    target_layer=target_layer,
                    input_tensor=input_tensor.clone(),
                    raw_pil_image=pil_image,
                    target_class_idx=top_class_idx
                )
                if not gradcam_b64:
                    gradcam_status = "generation_failed"
            except Exception as e:
                print(f"Grad-CAM execution exception: {e}")
                gradcam_status = f"error: {str(e)[:50]}"
        gradcam_end = time.perf_counter()
        gradcam_ms = (gradcam_end - gradcam_start) * 1000.0

        # Stage 6: Gemini AI Contextual Explanation (Cached / Fallback)
        gemini_start = time.perf_counter()
        explanation = gemini_service.generate_explanation(
            plant=disease_info.plant if disease_info else top_class_id.split("_")[0].capitalize(),
            predicted_condition=disease_info.name if disease_info else top_class_id.replace("_", " ").title(),
            confidence_percent=calibrated_top_prob * 100.0,
            disease_info=disease_info,
            state=pred_state
        )
        gemini_end = time.perf_counter()
        gemini_ms = (gemini_end - gemini_start) * 1000.0

        # Stage 7: Optional Multi-Model Comparison
        comparison_result = None
        if enable_model_comparison:
            try:
                comparison_result = model_registry.run_model_comparison(pil_image)
            except Exception as e:
                print(f"Multi-model comparison exception: {e}")

        total_end = time.perf_counter()
        total_ms = (total_end - total_start) * 1000.0

        perf = PerformanceMetrics(
            image_validation_ms=round(validation_ms, 2),
            preprocessing_ms=round(preprocessing_ms, 2),
            model_inference_ms=round(inference_ms, 2),
            gradcam_ms=round(gradcam_ms, 2),
            disease_metadata_lookup_ms=round(lookup_ms, 2),
            gemini_ms=round(gemini_ms, 2),
            total_request_ms=round(total_ms, 2)
        )

        audit = PredictionAudit(
            request_id=req_id,
            model_id=model_wrapper.model_id,
            model_version=model_wrapper.version,
            prediction_state=pred_state,
            raw_confidence=round(raw_top_prob, 4),
            calibrated_confidence=round(calibrated_top_prob, 4),
            temperature_applied=round(temp, 3),
            entropy=round(entropy, 4),
            top1_top2_margin=round(top_margin, 4),
            suitability_score=quality_result.suitability_score,
            validator_status=quality_result.status,
            reason_code=quality_result.reason_code,
            gradcam_status=gradcam_status,
            gemini_status="gemini_api" if (explanation and explanation.powered_by_gemini) else "curated_synthesis",
            performance_metrics=perf
        )

        display_confidence = calibrated_top_prob if settings.ENABLE_TEMPERATURE_CALIBRATION else raw_top_prob

        prediction_result = PredictionResult(
            class_id=top_class_id,
            name=disease_info.name if disease_info else top_class_id.replace("_", " ").title(),
            scientific_name=disease_info.scientific_name if disease_info else None,
            plant=disease_info.plant if disease_info else top_class_id.split("_")[0].capitalize(),
            state=pred_state,
            confidence=round(display_confidence, 4),
            raw_confidence=round(raw_top_prob, 4),
            calibrated_confidence=round(calibrated_top_prob, 4),
            confidence_percent=round(display_confidence * 100.0, 1),
            confidence_level=conf_level,
            entropy=round(entropy, 4),
            top1_top2_margin=round(top_margin, 4),
            is_healthy=is_healthy,
            status_message=status_msg,
            top_candidates=candidates
        )

        return AnalysisResponse(
            success=True,
            is_plant=True,
            prediction=prediction_result,
            model=model_wrapper.to_metadata_schema(is_default=(model_wrapper.model_id == model_registry.default_model_id)),
            quality=quality_result,
            disease=disease_info,
            non_plant_details=None,
            explanation=explanation,
            gradcam_heatmap_base64=gradcam_b64,
            comparison=comparison_result,
            audit=audit,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

inference_service = InferenceService()
