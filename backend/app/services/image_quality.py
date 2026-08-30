"""
Image Suitability & Quality Assessment Service for PlantCare
Evaluates botanical plant presence, resolution, blur, lighting, contrast, multi-leaf clustering,
and subject framing focus with targeted diagnostic reason codes.
"""

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Tuple, List, Dict, Any, Optional

from app.core.config import settings
from app.schemas.analysis import QualityCheckResult, QualityMetrics
from app.services.plant_validator import plant_validator

class ImageQualityService:
    def __init__(self):
        self.blur_threshold = settings.QUALITY_BLUR_THRESHOLD
        self.min_brightness = settings.QUALITY_MIN_BRIGHTNESS
        self.max_brightness = settings.QUALITY_MAX_BRIGHTNESS
        self.min_contrast = settings.QUALITY_MIN_CONTRAST
        self.min_veg_ratio = settings.QUALITY_MIN_VEGETATION_RATIO
        self.min_resolution = settings.QUALITY_MIN_RESOLUTION

    def evaluate_image(self, image_bytes: bytes) -> QualityCheckResult:
        """
        Runs comprehensive image quality and plant presence suitability checks.
        """
        # 1. First & Foremost: Validate Plant vs. Non-Plant + Multi-leaf & Focus
        plant_val = plant_validator.validate_image(image_bytes)

        # Load image with PIL and OpenCV
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        width, height = pil_image.size

        cv_img = np.array(pil_image)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)

        # 2. Optical Quality Metrics
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        veg_ratio = float(plant_val.foliage_ratio / 100.0)

        metrics = QualityMetrics(
            width=width,
            height=height,
            blur_score=round(laplacian_var, 2),
            brightness=round(mean_brightness, 2),
            contrast=round(contrast, 2),
            vegetation_ratio=round(plant_val.foliage_ratio, 2),
            estimated_leaf_count=plant_val.leaf_count_estimate,
            background_ratio=round(plant_val.background_ratio * 100.0, 2)
        )

        issues: List[str] = []
        positive_indicators: List[str] = []
        warnings_list: List[str] = list(plant_val.warnings)
        targeted_guidance_items: List[str] = []

        # ---------------------------------------------------------
        # Case A: Out-of-Domain Non-Plant Image
        # ---------------------------------------------------------
        if not plant_val.is_plant:
            issues.append(f"Non-plant image detected: {plant_val.detected_subject}.")
            if plant_val.rejection_reason:
                issues.append(plant_val.rejection_reason)

            guidance = f"PlantCare is designed strictly for plant pathology. The uploaded image appears to be a '{plant_val.detected_subject}'. Please upload a clear photo of a plant leaf or crop."

            return QualityCheckResult(
                is_suitable=False,
                suitability_score=0.0,
                status="rejected",
                is_plant=False,
                detected_subject=plant_val.detected_subject,
                subject_category=plant_val.subject_category,
                plant_confidence=plant_val.plant_confidence,
                reason_code="NON_PLANT_OBJECT",
                warnings=warnings_list,
                has_multiple_leaves=False,
                leaf_focus_status="non_plant",
                issues=issues,
                positive_indicators=[],
                metrics=metrics,
                guidance=guidance
            )

        # ---------------------------------------------------------
        # Case B: Valid Plant Specimen - Evaluate Optical Quality & Framing
        # ---------------------------------------------------------
        positive_indicators.append(f"Plant specimen verified ({plant_val.detected_subject})")
        score = 100.0
        primary_reason_code = "SUITABLE_PLANT"

        # Check resolution
        if width < self.min_resolution or height < self.min_resolution:
            issues.append(f"Image resolution low ({width}x{height}px). Minimum recommended is {self.min_resolution}x{self.min_resolution}px.")
            targeted_guidance_items.append("Upload a higher-resolution image.")
            warnings_list.append("LOW_RESOLUTION")
            primary_reason_code = "LOW_RESOLUTION"
            score -= 30.0
        else:
            positive_indicators.append(f"Resolution sufficient ({width}x{height}px)")

        # Check blur
        if laplacian_var < self.blur_threshold:
            issues.append("Image appears blurry or out of focus. Fine lesion details may be obscured.")
            targeted_guidance_items.append("Hold the camera steady and make sure the leaf is in focus.")
            warnings_list.append("BLURRY")
            primary_reason_code = "BLURRY"
            score -= 25.0
        else:
            positive_indicators.append("Focus is sharp and clear")

        # Check brightness
        if mean_brightness < self.min_brightness:
            issues.append("Image is too dark. Low lighting may conceal fungal textures.")
            targeted_guidance_items.append("Move to brighter natural lighting.")
            warnings_list.append("TOO_DARK")
            primary_reason_code = "TOO_DARK"
            score -= 25.0
        elif mean_brightness > self.max_brightness:
            issues.append("Image is overexposed or too bright. Glare may wash out leaf patterns.")
            targeted_guidance_items.append("Move out of direct glare or harsh sunlight.")
            warnings_list.append("TOO_BRIGHT")
            primary_reason_code = "TOO_BRIGHT"
            score -= 25.0
        else:
            positive_indicators.append("Lighting and exposure are well-balanced")

        # Check contrast
        if contrast < self.min_contrast:
            issues.append("Image has low contrast. Leaf features might blend into background.")
            targeted_guidance_items.append("Ensure distinct contrast between leaf and background.")
            warnings_list.append("POOR_CONTRAST")
            score -= 15.0
        else:
            positive_indicators.append("Contrast is distinct")

        # Check vegetation / foliage coverage
        if veg_ratio < self.min_veg_ratio:
            issues.append("Leaf occupies a small portion of the frame. Move closer to the leaf.")
            targeted_guidance_items.append("Move closer so the leaf occupies more of the image.")
            if "LEAF_TOO_SMALL" not in warnings_list:
                warnings_list.append("LEAF_TOO_SMALL")
            primary_reason_code = "LEAF_TOO_SMALL"
            score -= 20.0
        else:
            positive_indicators.append(f"Foliage coverage strong ({plant_val.foliage_ratio}%)")

        # Check multiple leaves
        if plant_val.has_multiple_leaves:
            issues.append(f"Multiple leaves detected (~{plant_val.leaf_count_estimate} distinct subjects).")
            targeted_guidance_items.append("Photograph one leaf at a time for a more reliable result.")
            if "MULTIPLE_LEAVES" not in warnings_list:
                warnings_list.append("MULTIPLE_LEAVES")
            primary_reason_code = "MULTIPLE_LEAVES"
            score -= 15.0

        # Check partial leaf / obstruction
        if plant_val.leaf_focus_status == "partial_leaf":
            issues.append("Leaf appears partially outside the camera frame.")
            targeted_guidance_items.append("Make sure the entire leaf is clearly visible and not covered.")
            if "PARTIAL_LEAF" not in warnings_list:
                warnings_list.append("PARTIAL_LEAF")
            score -= 10.0

        score = max(0.0, min(100.0, score))

        # Status determination & comprehensive guidance
        if score >= 70.0 and len(issues) == 0:
            status = "suitable"
            is_suitable = True
            primary_reason_code = "SUITABLE_PLANT"
            guidance = "Your image meets all quality standards for reliable AI diagnosis."
        elif score >= 40.0:
            status = "warning"
            is_suitable = True
            if primary_reason_code == "SUITABLE_PLANT" and len(warnings_list) > 0:
                primary_reason_code = warnings_list[0]
            guidance = " ".join(targeted_guidance_items) if targeted_guidance_items else "Image quality is sub-optimal. You can continue or take a clearer photo."
        else:
            status = "rejected"
            is_suitable = False
            guidance = " ".join(targeted_guidance_items) if targeted_guidance_items else "Image quality is insufficient for reliable plant pathology analysis."

        return QualityCheckResult(
            is_suitable=is_suitable,
            suitability_score=round(score, 1),
            status=status,
            is_plant=True,
            detected_subject=plant_val.detected_subject,
            subject_category=plant_val.subject_category,
            plant_confidence=plant_val.plant_confidence,
            reason_code=primary_reason_code,
            warnings=list(set(warnings_list)),
            has_multiple_leaves=plant_val.has_multiple_leaves,
            leaf_focus_status=plant_val.leaf_focus_status,
            issues=issues,
            positive_indicators=positive_indicators,
            metrics=metrics,
            guidance=guidance
        )

image_quality_service = ImageQualityService()
